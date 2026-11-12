"""Measure what (played back at the video frame rate) the whole playback pipeline costs: decode the stream -> (upscale) -> full-res frame.
Run one method per fresh process so memory readings don't mix:  python resources.py <method>
Methods: native, native_vp120, none, bicubic, lanczos, fastsrgan, fastsrgan_vp120"""
import json, os, re, resource, subprocess, sys, threading, time
import cv2, numpy as np

method = sys.argv[1]
FRAMES = 150
OUT = "out"

def rss_mb():  # current peak resident memory of this process
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024

def gpu_stats():
    """Apple GPU utilisation % and GPU-side memory in use, from ioreg (no sudo needed)."""
    try:
        s = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
                           capture_output=True, text=True, timeout=2).stdout
        u = re.search(r'"Device Utilization %"=(\d+)', s)
        m = re.search(r'"In use system memory"=(\d+)', s)
        return (int(u.group(1)) if u else None, int(m.group(1)) / 2**20 if m else None)
    except Exception:
        return (None, None)

class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.util, self.mem, self.stop = [], [], False
    def run(self):
        while not self.stop:
            u, m = gpu_stats()
            if u is not None: self.util.append(u)
            if m is not None: self.mem.append(m)
            time.sleep(0.1)

# ---- set up the pipeline
src = {"native": f"{OUT}/reference.mp4", "native_vp120": f"{OUT}/stream_vp120.mp4"}.get(method, f"{OUT}/lowres.mp4")
up, torch = None, None
if method == "native_vp120":
    import bench
    _c = cv2.VideoCapture(f"{OUT}/reference.mp4"); FW, FH = int(_c.get(3)), int(_c.get(4)); _c.release()
    bx, by, _, _ = bench.viewport_box(FW, FH, 120)
    def up(f):
        full = np.zeros((FH, FW, 3), np.uint8); full[by:by + f.shape[0], bx:bx + f.shape[1]] = f; return full
elif method in ("none", "bicubic", "lanczos"):
    interp = {"none": cv2.INTER_NEAREST, "bicubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}[method]
    up = lambda f: cv2.resize(f, (f.shape[1] * 4, f.shape[0] * 4), interpolation=interp)
elif method.startswith("fastsrgan"):
    import torch
    import bench
    sr = bench.FastSRGANUpscaler("Fast-SRGAN", "auto", 0, True)
    up = bench.ViewportUpscaler(sr, 120, 4) if method.endswith("vp120") else sr
rss_after_setup = rss_mb()

# idle GPU reading before we start
idle = [gpu_stats() for _ in range(5)]
idle_util = np.mean([u for u, _ in idle if u is not None]) if any(u is not None for u, _ in idle) else None
idle_gmem = np.mean([m for _, m in idle if m is not None]) if any(m is not None for _, m in idle) else None

# ---- run
cap = cv2.VideoCapture(src)
fps_video = cap.get(cv2.CAP_PROP_FPS) or 25
samp = Sampler(); samp.start()
r0, t0 = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter()
dec_t, up_t, n, mps_peak, shape = 0.0, 0.0, 0, 0.0, None
while n < FRAMES:
    a = time.perf_counter()
    ok, f = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
    b = time.perf_counter(); dec_t += b - a
    out = up(f) if up else f
    up_t += time.perf_counter() - b
    shape = out.shape
    if torch is not None:
        mps_peak = max(mps_peak, bench.gpu_mem_mb(torch, sr.dev.type) or 0.0)
    n += 1
    # play at the video's frame rate, like a real player (sleep if we're ahead)
    ahead = t0 + n / fps_video - time.perf_counter()
    if ahead > 0: time.sleep(ahead)
wall = time.perf_counter() - t0
r1 = resource.getrusage(resource.RUSAGE_SELF)
samp.stop = True; samp.join()
cpu_s = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)

res = {
    "method": method,
    "stream_file": src, "stream_resolution": f"{int(cap.get(3))}x{int(cap.get(4))}",
    "output_resolution": f"{shape[1]}x{shape[0]}",
    "speed": {"ms_per_frame_busy": 1000 * (dec_t + up_t) / n, "decode_ms": 1000 * dec_t / n, "upscale_ms": 1000 * up_t / n,
              "fps": n / wall, "realtime": (dec_t + up_t) / n <= 1 / fps_video},
    "cpu": {"avg_cores_busy_during_playback": cpu_s / wall, "cpu_ms_per_frame": 1000 * cpu_s / n},
    "ram": {"peak_mb": rss_mb(), "after_setup_mb": rss_after_setup},
    "gpu": {"mps_driver_peak_mb": mps_peak or 0.0,
            "util_avg_pct": float(np.mean(samp.util)) if samp.util else None,
            "util_max_pct": float(np.max(samp.util)) if samp.util else None,
            "util_idle_pct": idle_util,
            "gpu_mem_in_use_mb": float(np.max(samp.mem)) if samp.mem else None,
            "gpu_mem_idle_mb": idle_gmem},
}
print(json.dumps(res))
os.makedirs("resources", exist_ok=True)
json.dump(res, open(f"resources/{method}.json", "w"), indent=1)
