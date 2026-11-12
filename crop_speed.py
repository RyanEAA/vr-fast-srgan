"""Time Fast-SRGAN on crop sizes matching what a VR viewer actually sees."""
import sys, time, json, re
from types import SimpleNamespace
import torch
sys.path.insert(0, "Fast-SRGAN")
from model import Generator
y = open("Fast-SRGAN/configs/config.yaml").read(); g = y[y.index("generator:"):]
cfg = SimpleNamespace(n_filters=int(re.search(r"n_filters:\s*(\d+)", g).group(1)),
                      n_layers=int(re.search(r"n_layers:\s*(\d+)", g).group(1)))
w = {k.replace("_orig_mod.", ""): v for k, v in torch.load("Fast-SRGAN/models/model.pt", map_location="cpu").items()}
import bench
dev = torch.device(bench.pick_device())
# (label, low-res input w, h). Low-res = source / 4. Pixels per degree of an equirect = width/360.
cases = []
for src, W in [("1080p clip (this test)", 1920), ("4K 360 video", 3840), ("8K 360 video", 7680)]:
    lw = W // 4; ppd = lw / 360
    if W <= 1920:  # 4K/8K full frames need several GB per activation and froze the Mac
        cases.append((f"{src}: full frame", lw, lw // 2))
    cases += [(f"{src}: viewport 120x120 deg (90 deg FOV + head-motion margin)", round(120 * ppd), round(120 * ppd)),
              (f"{src}: viewport 90x90 deg", round(90 * ppd), round(90 * ppd)),
              (f"{src}: fovea 20x20 deg (eye tracking)", round(20 * ppd), round(20 * ppd))]
res = []
for half in (False, True):
    if half and dev.type == "cpu": continue
    dt = torch.float16 if half else torch.float32
    m = Generator(cfg); m.load_state_dict(w); m.eval().to(dev, dt)
    for label, iw, ih in cases:
        # Big crops are run as 160x160 tiles (how a real viewer would have to do it) to stay
        # within memory; time = time per tile x number of tiles.
        tiles = 1
        if iw * ih * 16 > 2_600_000:
            tiles = -(-iw // 160) * -(-ih // 160)
            iw_run, ih_run = 160, 160
        else:
            iw_run, ih_run = iw, ih
        x = torch.rand(1, 3, ih_run, iw_run, device=dev, dtype=dt) * 2 - 1
        with torch.no_grad():
            for _ in range(3): m(x)
            bench.gpu_sync(torch, dev.type)
            n = 10 if iw * ih > 200000 else 30
            t = time.perf_counter()
            for _ in range(n): m(x)
            bench.gpu_sync(torch, dev.type)
        ms = (time.perf_counter() - t) / n * 1000 * tiles
        mem = bench.gpu_mem_mb(torch, dev.type) or 0
        r = dict(gpu_mem_mb=round(mem), tiles=tiles, case=label, fp16=half, input=f"{iw}x{ih}", output=f"{iw*4}x{ih*4}", ms=round(ms, 1), fps=round(1000 / ms, 1),
                 out_mpix_per_s=round(iw * ih * 16 / ms / 1000, 1))
        res.append(r); print(json.dumps(r), flush=True)
        del x; bench.gpu_free(torch, dev.type)
json.dump(res, open("crop_speed.json", "w"), indent=1)
print("CROP_DONE")
