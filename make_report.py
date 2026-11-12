"""Builds results/RESULTS.md from the measurement files (out/, resources/, crop_speed.json)
and copies the raw JSON into results/data/ so the numbers are versioned with the repo."""
import glob, json, os, platform, shutil, subprocess, sys

os.makedirs("results/data", exist_ok=True)
J = lambda p: json.load(open(p)) if os.path.exists(p) else None
f = lambda x, d=1: "—" if x is None else f"{x:.{d}f}"

m = {k: J(f"out/metrics_{k}.json") for k in
     ["none", "bicubic", "lanczos", "fastsrgan", "fastsrgan_fp16", "fastsrgan_fp16_vp120", "native_vp120", "reference_inview"]}
r = {k: J(f"resources/{k}.json") for k in
     ["native", "native_vp120", "none", "bicubic", "lanczos", "fastsrgan", "fastsrgan_vp120"]}
crop = J("crop_speed.json")

def sysinfo():
    cpu = platform.processor()
    if sys.platform == "darwin":
        try: cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        except Exception: pass
    try:
        import torch; tv = torch.__version__
    except Exception: tv = "?"
    return f"{platform.system()} {platform.release()} · {cpu} · Python {platform.python_version()} · torch {tv}"

full_kbps = (m["reference_inview"] or {}).get("bandwidth", {}).get("lowres_stream_kbps")
low_kbps = (m["bicubic"] or {}).get("bandwidth", {}).get("lowres_stream_kbps")
nat = (m["bicubic"] or {}).get("native_res_at_same_bitrate", {})
gbh = lambda k: None if k is None else k * 3600 / 8 / 1e6

rows = [  # name, metrics key, resources key, stream kbps
    ("Full-res stream, whole 360° (baseline)", "reference_inview", "native", full_kbps),
    ("Full-res stream, visible region only", "native_vp120", "native_vp120", (m["native_vp120"] or {}).get("bandwidth", {}).get("lowres_stream_kbps")),
    ("Low-res stream + no upscaling", "none", "none", low_kbps),
    ("Low-res stream + bicubic", "bicubic", "bicubic", low_kbps),
    ("Low-res stream + Lanczos", "lanczos", "lanczos", low_kbps),
    ("Low-res stream + Fast-SRGAN (fp32)", "fastsrgan", None, low_kbps),
    ("Low-res stream + Fast-SRGAN (fp16)", "fastsrgan_fp16", "fastsrgan", low_kbps),
    ("Low-res stream + Fast-SRGAN on visible region (fp16)", "fastsrgan_fp16_vp120", "fastsrgan_vp120", low_kbps),
]

L = ["# Results", "", f"Machine: {sysinfo()}  ", "Test clip: `clips/pano.mp4` (three.js sample, 1920×1080, 7.5 s, 25 fps). "
     "Low-res stream = ¼ width and height. All streams H.264 CRF 23 unless noted.", ""]

L += ["## Bandwidth, speed and quality", "",
      "Quality is measured on the **visible 120°×120° region** against the original video (higher PSNR/SSIM is better, lower flicker is better). "
      "Time per frame is the upscaling step only (decode time for the two full-res streams). Real time = under 40 ms (25 fps).", "",
      "| Technique | Stream kbps | GB/hour | ms/frame | Real time | PSNR dB | SSIM | Flicker |",
      "|---|---:|---:|---:|:---:|---:|---:|---:|"]
for name, mk, rk, kb in rows:
    x = m.get(mk) or {}; s = x.get("speed", {}); q = x.get("quality_in_view") or {}
    L.append(f"| {name} | {f(kb, 0)} | {f(gbh(kb), 2)} | {f(s.get('p50_ms'))} | {'✓' if s.get('realtime_at_video_fps') else '✗'} | "
             f"{f(q.get('psnr_db'))} | {f(q.get('ssim'), 3)} | {f(q.get('flicker_mean_abs'), 2)} |")
if nat:
    L += ["", f"Full-res stream squeezed to the low-res bitrate ({f(nat.get('kbps'), 0)} kbps): whole-frame PSNR {f(nat.get('psnr_db'))} dB, "
          f"vs {f((m['bicubic'] or {}).get('quality_vs_original', {}).get('psnr_db'))} dB for low-res + bicubic at the same bandwidth."]

L += ["", "## Resources while playing at 25 fps", "",
      "One fresh process per technique: decode the stream → upscale → full-res frame. GPU numbers are above the idle level "
      "(Apple GPU via `ioreg`; not available on other platforms).", "",
      "| Technique | CPU (% of 1 core) | RAM peak MB | GPU memory added MB | GPU busy % |", "|---|---:|---:|---:|---:|"]
for name, mk, rk, kb in rows:
    x = r.get(rk) if rk else None
    if not x: continue
    g = x["gpu"]
    gm = None if g.get("gpu_mem_in_use_mb") is None else max(0, g["gpu_mem_in_use_mb"] - (g.get("gpu_mem_idle_mb") or 0))
    gu = None if g.get("util_avg_pct") is None else max(0, g["util_avg_pct"] - (g.get("util_idle_pct") or 0))
    if gm is None and g.get("mps_driver_peak_mb"): gm = g["mps_driver_peak_mb"]
    L.append(f"| {name} | {f(x['cpu']['avg_cores_busy_during_playback'] * 100, 0)} | {f(x['ram']['peak_mb'], 0)} | {f(gm, 0)} | {f(gu, 0)} |")

if crop:
    L += ["", "## Fast-SRGAN time vs. how much of the view is upscaled", "",
          "Large crops run as 160×160 low-res tiles to keep GPU memory around 1 GB (whole 4K/8K frames in one pass exhausted 16 GB).", "",
          "| Source | Region | Input → output | fp32 ms | fp16 ms | fp16 fps |", "|---|---|---|---:|---:|---:|"]
    by = {}
    for c in crop: by.setdefault(c["case"], {})["fp16" if c["fp16"] else "fp32"] = c
    for case, d in by.items():
        src, reg = case.split(": ", 1); a = d.get("fp16") or d.get("fp32")
        L.append(f"| {src} | {reg} | {a['input']} → {a['output']} | {f(d.get('fp32', {}).get('ms'))} | {f(d.get('fp16', {}).get('ms'))} | {f(d.get('fp16', {}).get('fps'))} |")

open("results/RESULTS.md", "w").write("\n".join(L) + "\n")
for p in glob.glob("out/metrics_*.json") + glob.glob("resources/*.json") + ["crop_speed.json"]:
    if os.path.exists(p): shutil.copy(p, "results/data/" + p.replace("/", "_").replace("out_", "").replace("resources_", "resources_"))
for img in ["viewport_compare.png", "compare_frame75.png"]:
    if os.path.exists(img): shutil.copy(img, f"results/{img}")
print("wrote results/RESULTS.md")
