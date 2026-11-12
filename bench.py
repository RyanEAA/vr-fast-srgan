#!/usr/bin/env python3
"""
Benchmark "stream low-res, upscale on device" for 360/VR video.

Pipeline (simulates a real stream, codec artifacts included):
  original equirect video
    -> downscale by --scale (INTER_AREA)
    -> encode low-res with H.264 at --crf     (this is the "streamed" file; its size = bandwidth)
    -> decode it
    -> upscale with bicubic / lanczos / Fast-SRGAN
    -> compare against the original frames

Also encodes the original at the same CRF so you can compare bandwidth, and
optionally a "same bitrate" native-res encode (--match-bitrate) which is the real
competitor: is SR(low-res @ B kbps) better than native-res @ B kbps?

Outputs in --out:
  lowres.mp4, <upscaler>.mp4, reference.mp4, [native_same_bitrate.mp4], metrics_<upscaler>.json

Usage:
  python bench.py --video clip.mp4 --upscaler bicubic
  python bench.py --video clip.mp4 --upscaler fastsrgan --half
"""
import argparse, json, os, resource, subprocess, sys, time
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------- devices
def pick_device(pref="auto"):
    """'auto' -> Apple GPU (mps) if available, else NVIDIA (cuda), else cpu."""
    import torch
    if pref != "auto": return pref
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"


def gpu_sync(torch, dev_type):
    if dev_type == "mps": torch.mps.synchronize()
    elif dev_type == "cuda": torch.cuda.synchronize()


def gpu_mem_mb(torch, dev_type):
    """Memory the GPU driver holds for this process (None on cpu)."""
    if dev_type == "mps": return torch.mps.driver_allocated_memory() / 2**20
    if dev_type == "cuda": return torch.cuda.memory_reserved() / 2**20
    return None


def gpu_free(torch, dev_type):
    if dev_type == "mps": torch.mps.empty_cache()
    elif dev_type == "cuda": torch.cuda.empty_cache()


# ---------------------------------------------------------------- upscalers
class CVUpscaler:
    def __init__(self, scale, interp):
        self.scale, self.interp = scale, interp
        self.name = {cv2.INTER_NEAREST: "none", cv2.INTER_CUBIC: "bicubic", cv2.INTER_LANCZOS4: "lanczos"}[interp]

    def __call__(self, bgr):
        h, w = bgr.shape[:2]
        return cv2.resize(bgr, (w * self.scale, h * self.scale), interpolation=self.interp)

    def device_mem(self):
        return None


class FastSRGANUpscaler:
    """Adapter for the current (PyTorch) Fast-SRGAN repo: models/model.pt is a state_dict for
    model.Generator(n_filters=64, n_layers=8); input and output are in [-1, 1]."""

    def __init__(self, repo, device, tile=0, half=False):
        import torch
        from types import SimpleNamespace
        self.torch, self.name, self.tile = torch, "fastsrgan_fp16" if half else "fastsrgan", tile
        repo = Path(repo).expanduser()
        sys.path.insert(0, str(repo))
        from model import Generator
        cfg = SimpleNamespace(n_filters=64, n_layers=8)
        try:  # read the real values from the repo config if present
            import re
            y = (repo / "configs/config.yaml").read_text()
            g = y[y.index("generator:"):]
            cfg.n_filters = int(re.search(r"n_filters:\s*(\d+)", g).group(1))
            cfg.n_layers = int(re.search(r"n_layers:\s*(\d+)", g).group(1))
        except Exception:
            pass
        w = torch.load(repo / "models/model.pt", map_location="cpu")
        w = {k.replace("_orig_mod.", ""): v for k, v in w.items()}
        self.model = Generator(cfg)
        self.model.load_state_dict(w)
        self.dev = torch.device(pick_device(device))
        if self.dev.type == "cpu" and half:
            print("[fastsrgan] fp16 isn't useful on cpu, using fp32"); half = False
        self.dtype = torch.float16 if half else torch.float32
        self.model.eval().to(self.dev, self.dtype)
        self.params = sum(p.numel() for p in self.model.parameters())
        print(f"[fastsrgan] device={self.dev} dtype={self.dtype} params={self.params:,} cfg={cfg}")

    def _sync(self):
        gpu_sync(self.torch, self.dev.type)

    def _run(self, rgb):  # HxWx3 uint8 -> HxWx3 uint8
        t = self.torch
        with t.no_grad():
            x = t.from_numpy(rgb).to(self.dev).permute(2, 0, 1)[None].to(self.dtype) / 127.5 - 1.0
            y = self.model(x)
            y = ((y[0].permute(1, 2, 0).float() + 1.0) * 127.5).clamp(0, 255).round().to(t.uint8)
            self._sync()
            return y.cpu().numpy()

    def __call__(self, bgr):
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        if self.tile:  # note: InstanceNorm makes tiled output differ slightly from whole-frame
            T, (H, W) = self.tile, rgb.shape[:2]
            out = np.zeros((H * 4, W * 4, 3), np.uint8)
            for y0 in range(0, H, T):
                for x0 in range(0, W, T):
                    o = self._run(np.ascontiguousarray(rgb[y0:y0 + T, x0:x0 + T]))
                    out[y0 * 4:y0 * 4 + o.shape[0], x0 * 4:x0 * 4 + o.shape[1]] = o
        else:
            out = self._run(rgb)
        return np.ascontiguousarray(out[:, :, ::-1])

    def device_mem(self):
        if self.dev.type in ("mps", "cuda"):
            return {"device": self.dev.type,
                    "mps_driver_mb": gpu_mem_mb(self.torch, self.dev.type),
                    "params": self.params}
        return {"params": self.params}


class ViewportUpscaler:
    """Fast-SRGAN only inside a centred viewport of `deg` x `deg` degrees of the equirect frame,
    bicubic everywhere else. Simulates upscaling just what the viewer is looking at."""

    def __init__(self, sr, deg, scale):
        self.sr, self.deg, self.scale = sr, deg, scale
        self.base = CVUpscaler(scale, cv2.INTER_CUBIC)
        self.name = f"{sr.name}_vp{deg}"

    def box(self, w, h):
        return viewport_box(w, h, self.deg)

    def __call__(self, bgr):
        out = self.base(bgr)
        h, w = bgr.shape[:2]
        x0, y0, cw, ch = self.box(w, h)
        s = self.scale
        out[y0 * s:(y0 + ch) * s, x0 * s:(x0 + cw) * s] = self.sr(np.ascontiguousarray(bgr[y0:y0 + ch, x0:x0 + cw]))
        return out

    def device_mem(self):
        return self.sr.device_mem()


def viewport_box(w, h, deg):
    """(x0, y0, width, height) of a deg x deg region centred in an equirect frame of size w x h."""
    cw, ch = round(deg / 360 * w), round(min(deg, 180) / 180 * h)
    return (w - cw) // 2, (h - ch) // 2, cw, ch


# ---------------------------------------------------------------- helpers
def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def encode(src, dst, crf=None, kbps=None, scale=None, frames=None):
    vf = [f"scale=iw/{scale}:ih/{scale}:flags=area"] if scale else []
    rc = ["-crf", str(crf)] if crf is not None else ["-b:v", f"{kbps}k", "-maxrate", f"{kbps}k", "-bufsize", f"{2*kbps}k"]
    ffmpeg("-i", str(src), *(["-frames:v", str(frames)] if frames else []),
           *(["-vf", ",".join(vf)] if vf else []), "-an", "-c:v", "libx264", "-preset", "medium",
           "-pix_fmt", "yuv420p", *rc, str(dst))


def kbps_of(path, fps, n):
    return os.path.getsize(path) * 8 / 1000 / (n / fps)


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def ssim_gray(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    g = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    ma, mb = g(a), g(b)
    va, vb, cov = g(a * a) - ma ** 2, g(b * b) - mb ** 2, g(a * b) - ma * mb
    return float(np.mean(((2 * ma * mb + C1) * (2 * cov + C2)) / ((ma ** 2 + mb ** 2 + C1) * (va + vb + C2))))


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024  # bytes on macOS, KB on Linux


def pct(x, p):
    return float(np.percentile(x, p)) if len(x) else None


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--scale", type=int, default=4, help="downscale factor; Fast-SRGAN is 4x")
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--frames", type=int, default=150)
    # "none" = no upscaling: each low-res pixel is just repeated 4x4 (nearest neighbour)
    ap.add_argument("--upscaler", choices=["none", "bicubic", "lanczos", "fastsrgan"], default="bicubic")
    ap.add_argument("--repo", default="Fast-SRGAN", help="path to the cloned Fast-SRGAN repo")
    ap.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    ap.add_argument("--tile", type=int, default=0, help="low-res tile size, 0 = whole frame")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--half", action="store_true", help="fp16 inference")
    ap.add_argument("--viewport", type=int, default=0,
                    help="if set (degrees), run the upscaler only inside a centred viewport; bicubic elsewhere")
    ap.add_argument("--eval-region", type=int, default=120,
                    help="also score quality inside a centred region of this many degrees (0 = off)")
    ap.add_argument("--match-bitrate", action="store_true",
                    help="also encode native-res at the low-res stream's bitrate and score it")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = min(a.frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or a.frames)
    cap.release()

    # Reference frames trimmed so the dimensions divide evenly by scale.
    print("encoding streams...")
    low, ref = out / "lowres.mp4", out / "reference.mp4"
    encode(a.video, low, crf=a.crf, scale=a.scale, frames=n)
    encode(a.video, ref, crf=a.crf, frames=n)
    low_kbps, ref_kbps = kbps_of(low, fps, n), kbps_of(ref, fps, n)

    rss_before = peak_rss_mb()
    t0 = time.perf_counter()
    up = (FastSRGANUpscaler(a.repo, a.device, a.tile, a.half) if a.upscaler == "fastsrgan"
          else CVUpscaler(a.scale, {"none": cv2.INTER_NEAREST, "bicubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}[a.upscaler]))
    if a.viewport:
        up = ViewportUpscaler(up, a.viewport, a.scale)
    load_s = time.perf_counter() - t0
    rss_model = peak_rss_mb()

    gt_cap, lr_cap = cv2.VideoCapture(a.video), cv2.VideoCapture(str(low))
    writer, lat, ps, ss, flick = None, [], [], [], []
    rps, rss_, rflick = [], [], []
    prev_rsr = prev_rgt = None
    prev_sr = prev_gt = None
    first_frame_s = None
    for i in range(n):
        ok1, gt = gt_cap.read(); ok2, lr = lr_cap.read()
        if not (ok1 and ok2):
            break
        t = time.perf_counter()
        sr = up(lr)
        dt = time.perf_counter() - t
        if i == 0:
            first_frame_s = dt
        if i >= a.warmup:
            lat.append(dt)
        gt = cv2.resize(gt, (sr.shape[1], sr.shape[0]), interpolation=cv2.INTER_AREA) if gt.shape != sr.shape else gt
        ps.append(psnr(sr, gt)); ss.append(ssim_gray(sr, gt))
        if prev_sr is not None:  # temporal flicker: change in SR that the ground truth didn't have
            d = (sr.astype(np.int16) - prev_sr) - (gt.astype(np.int16) - prev_gt)
            flick.append(float(np.mean(np.abs(d))))
        prev_sr, prev_gt = sr.astype(np.int16), gt.astype(np.int16)
        if a.eval_region:
            x0, y0, cw, ch = viewport_box(lr.shape[1], lr.shape[0], a.eval_region)
            k = sr.shape[1] // lr.shape[1]
            rs, rg = sr[y0*k:(y0+ch)*k, x0*k:(x0+cw)*k], gt[y0*k:(y0+ch)*k, x0*k:(x0+cw)*k]
            rps.append(psnr(rs, rg)); rss_.append(ssim_gray(rs, rg))
            if prev_rsr is not None:
                d = (rs.astype(np.int16) - prev_rsr) - (rg.astype(np.int16) - prev_rgt)
                rflick.append(float(np.mean(np.abs(d))))
            prev_rsr, prev_rgt = rs.astype(np.int16), rg.astype(np.int16)
        if writer is None:
            writer = cv2.VideoWriter(str(out / f"{up.name}_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (sr.shape[1], sr.shape[0]))
        writer.write(sr)
        if i % 25 == 0:
            print(f"  frame {i}/{n}  {dt*1000:.1f} ms  psnr {ps[-1]:.2f}")
    writer.release()
    # re-encode to H.264 at high quality so browsers can play it
    ffmpeg("-i", str(out / f"{up.name}_raw.mp4"), "-c:v", "libx264", "-crf", "12",
           "-pix_fmt", "yuv420p", str(out / f"{up.name}.mp4"))
    os.remove(out / f"{up.name}_raw.mp4")

    m = {
        "upscaler": up.name, "frames": len(ps), "video_fps": fps,
        "lowres_size": [int(lr.shape[1]), int(lr.shape[0])] if lr is not None else None,
        "output_size": [int(sr.shape[1]), int(sr.shape[0])],
        "speed": {
            "model_load_s": load_s,
            "first_frame_ms": first_frame_s * 1000,
            "p50_ms": pct(lat, 50) * 1000, "p95_ms": pct(lat, 95) * 1000, "max_ms": max(lat) * 1000,
            "throughput_fps": len(lat) / sum(lat),
            "realtime_at_video_fps": len(lat) / sum(lat) >= fps,
        },
        "memory": {
            "peak_rss_mb": peak_rss_mb(), "rss_added_by_model_mb": rss_model - rss_before,
            "device": up.device_mem(),
        },
        "bandwidth": {"lowres_stream_kbps": low_kbps, "native_stream_kbps_same_crf": ref_kbps,
                      "saving_pct": 100 * (1 - low_kbps / ref_kbps)},
        "quality_vs_original": {"psnr_db": float(np.mean(ps)), "ssim": float(np.mean(ss)),
                                "flicker_mean_abs": float(np.mean(flick)) if flick else None},
        "viewport_deg": a.viewport or None,
    }
    if a.eval_region:
        m["quality_in_view"] = {"region_deg": a.eval_region, "psnr_db": float(np.mean(rps)),
                                "ssim": float(np.mean(rss_)),
                                "flicker_mean_abs": float(np.mean(rflick)) if rflick else None}

    if a.match_bitrate:
        nat = out / "native_same_bitrate.mp4"
        encode(a.video, nat, kbps=max(50, int(low_kbps)), frames=n)
        g, v = cv2.VideoCapture(a.video), cv2.VideoCapture(str(nat))
        p2 = []
        for _ in range(len(ps)):
            ok1, x = g.read(); ok2, y = v.read()
            if not (ok1 and ok2): break
            x = cv2.resize(x, (sr.shape[1], sr.shape[0]), interpolation=cv2.INTER_AREA)
            y = cv2.resize(y, (sr.shape[1], sr.shape[0]), interpolation=cv2.INTER_AREA)
            p2.append(psnr(y, x))
        m["native_res_at_same_bitrate"] = {"kbps": kbps_of(nat, fps, n), "psnr_db": float(np.mean(p2))}

    (out / f"metrics_{up.name}.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
