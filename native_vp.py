"""Full-res stream of ONLY the visible 120x120 deg region (no upscaler), scored like the others.
Also scores the normal full-res stream on the same region, for comparison."""
import json, subprocess, time
import cv2, numpy as np
import bench

import sys
SRC, N, CRF, DEG = (sys.argv[1] if len(sys.argv) > 1 else "clips/pano.mp4"), 150, 23, 120
cap = cv2.VideoCapture(SRC); W, H = int(cap.get(3)), int(cap.get(4)); fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
x0, y0, cw, ch = bench.viewport_box(W, H, DEG)        # full-res region: same box as the other tests
cw -= cw % 2; ch -= ch % 2
stream = "out/stream_vp120.mp4"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", SRC, "-frames:v", str(N), "-vf", f"crop={cw}:{ch}:{x0}:{y0}",
                "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(CRF), "-pix_fmt", "yuv420p", stream], check=True)

def score(path, crop_from_full):
    g, v = cv2.VideoCapture(SRC), cv2.VideoCapture(path)
    ps, ss, fl, dec, prev = [], [], [], [], None
    for _ in range(N):
        ok1, gt = g.read(); t = time.perf_counter(); ok2, f = v.read(); dec.append(time.perf_counter() - t)
        if not (ok1 and ok2): break
        gt = gt[y0:y0 + ch, x0:x0 + cw]
        if crop_from_full: f = f[y0:y0 + ch, x0:x0 + cw]
        ps.append(bench.psnr(f, gt)); ss.append(bench.ssim_gray(f, gt))
        cur = (f.astype(np.int16), gt.astype(np.int16))
        if prev is not None: fl.append(float(np.mean(np.abs((cur[0] - prev[0]) - (cur[1] - prev[1])))))
        prev = cur
    return dict(psnr_db=float(np.mean(ps)), ssim=float(np.mean(ss)), flicker_mean_abs=float(np.mean(fl)),
                region_deg=DEG), 1000 * float(np.median(dec[5:]))

dur = N / fps
kbps = lambda p: __import__("os").path.getsize(p) * 8 / 1000 / dur
out = {}
for name, path, crop in [("reference", "out/reference.mp4", True), ("native_vp120", stream, False)]:
    q, dec_ms = score(path, crop)
    m = {"upscaler": name, "stream_resolution": "1920x1080" if crop else f"{cw}x{ch}",
         "speed": {"p50_ms": dec_ms, "throughput_fps": 1000 / dec_ms, "realtime_at_video_fps": dec_ms < 1000 / fps,
                   "note": "decode time only; no upscaling"},
         "bandwidth": {"lowres_stream_kbps": kbps(path), "stream_kb": __import__("os").path.getsize(path) / 1024},
         "quality_in_view": q}
    json.dump(m, open(f"out/metrics_{name}.json" if name != "reference" else "out/metrics_reference_inview.json", "w"), indent=1)
    out[name] = m; print(json.dumps(m))

# video for the viewer: the region pasted into a black 360 frame
v = cv2.VideoCapture(stream)
w = cv2.VideoWriter("out/native_vp120_raw.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
while True:
    ok, f = v.read()
    if not ok: break
    full = np.zeros((H, W, 3), np.uint8); full[y0:y0 + ch, x0:x0 + cw] = f; w.write(full)
w.release()
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "out/native_vp120_raw.mp4", "-c:v", "libx264", "-crf", "12",
                "-pix_fmt", "yuv420p", "out/native_vp120.mp4"], check=True)
print("NATIVE_VP_DONE")
