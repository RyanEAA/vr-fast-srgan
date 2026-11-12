"""Live pipeline + web server for the viewer.
Runs the selected technique for real (decode stream -> upscale) at the video frame rate on the selected video,
sends frames to the browser as MJPEG, and samples CPU / RAM / GPU every 0.5 s.
Videos: everything in clips/ (plus uploads) is prepared into prepared/<name>/ :
  ref.mp4 (full-res stream), low.mp4 (1/4 resolution stream), native.mp4 (full-res at the low-res bitrate),
  vp.mp4 (full-res stream of only the centred 120x120 deg region).
Open http://localhost:8766/viewer.html"""
import collections, gc, http.server, json, os, queue, re, resource, shutil, subprocess, sys, threading, time, urllib.parse
import cv2, numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
import bench
PORT, MAX_SECONDS, MAX_WIDTH = 8766, 15, 3840
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi")
os.makedirs("clips", exist_ok=True); os.makedirs("prepared", exist_ok=True)

TECH = {  # key: (which prepared stream, processing)
    "reference": ("ref", None), "native": ("native", None), "nvp": ("vp", "paste"),
    "none": ("low", "nearest"), "bicubic": ("low", "bicubic"), "lanczos": ("low", "lanczos"),
    "fastsrgan": ("low", "srgan32"), "fp16": ("low", "srgan16"), "vp": ("low", "srgan16_vp"),
}

# ---------------------------------------------------------------- video library
clips = {}            # name -> {status, step, error, w, h, seconds}
prep_q = queue.Queue()
def clip_name(fn): return re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.splitext(fn)[0])[:60] or "video"
def even(x): return int(x) // 2 * 2
def ff(*a): subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *a], check=True)

def scan():
    for fn in sorted(os.listdir("clips")):
        if not fn.lower().endswith(VIDEO_EXT): continue
        n = clip_name(fn)
        if n in clips: continue
        done = all(os.path.exists(f"prepared/{n}/{k}.mp4") for k in ("ref", "low", "native", "vp"))
        meta = {}
        if done and os.path.exists(f"prepared/{n}/meta.json"): meta = json.load(open(f"prepared/{n}/meta.json"))
        clips[n] = {"file": fn, "status": "ready" if done else "queued", "step": "", "error": None, **meta}
        if not done: prep_q.put(n)

def prepare(n):
    c = clips[n]; src = f"clips/{c['file']}"; d = f"prepared/{n}"; os.makedirs(d, exist_ok=True)
    try:
        c["status"] = "preparing"
        cap = cv2.VideoCapture(src); W0, H0 = int(cap.get(3)), int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30; nfr = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0; cap.release()
        if W0 == 0: raise RuntimeError("couldn't read this video")
        secs = min(MAX_SECONDS, nfr / fps if nfr else MAX_SECONDS)
        s = min(1.0, MAX_WIDTH / W0); W, H = even(W0 * s), even(H0 * s)
        lw, lh = even(W / 4), even(H / 4)
        x0, y0, cw, ch = bench.viewport_box(W, H, 120); cw, ch = even(cw), even(ch)
        if n == "pano" and os.path.exists("out/reference.mp4"):  # reuse the exact benchmark streams
            c["step"] = "copying benchmark streams"
            for a, b in [("reference", "ref"), ("lowres", "low"), ("native_same_bitrate", "native"), ("stream_vp120", "vp")]:
                shutil.copy(f"out/{a}.mp4", f"{d}/{b}.mp4")
        else:
            enc = ["-an", "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p"]
            c["step"] = "encoding full-res stream (1/4)"
            ff("-i", src, "-t", str(secs), "-vf", f"scale={W}:{H}:flags=area", *enc, "-crf", "23", f"{d}/ref.mp4")
            c["step"] = "encoding low-res stream (2/4)"
            ff("-i", src, "-t", str(secs), "-vf", f"scale={lw}:{lh}:flags=area", *enc, "-crf", "23", f"{d}/low.mp4")
            kbps = max(50, int(os.path.getsize(f"{d}/low.mp4") * 8 / 1000 / secs))
            c["step"] = "encoding full-res at low-res bitrate (3/4)"
            ff("-i", src, "-t", str(secs), "-vf", f"scale={W}:{H}:flags=area", *enc, "-b:v", f"{kbps}k",
               "-maxrate", f"{kbps}k", "-bufsize", f"{2 * kbps}k", f"{d}/native.mp4")
            c["step"] = "encoding visible-region stream (4/4)"
            ff("-i", src, "-t", str(secs), "-vf", f"scale={W}:{H}:flags=area,crop={cw}:{ch}:{x0}:{y0}", *enc, "-crf", "23", f"{d}/vp.mp4")
        err = subprocess.run(["ffmpeg", "-hide_banner", "-i", src], capture_output=True, text=True).stderr
        sm = re.search(r"SAR (\d+):(\d+)", err); sar = int(sm.group(1)) / int(sm.group(2)) if sm and int(sm.group(2)) else 1.0
        meta = {"w": W, "h": H, "seconds": round(secs, 1), "aspect_ok": abs(W * sar / H - 2) < 0.1, "downscaled_from": f"{W0}x{H0}" if s < 1 else None}
        json.dump(meta, open(f"{d}/meta.json", "w")); c.update(meta, status="ready", step="")
    except Exception as e:
        c.update(status="error", error=str(e)[:300], step="")

def prep_worker():
    while True:
        prepare(prep_q.get())

# ---------------------------------------------------------------- upscalers (Fast-SRGAN loaded on demand)
_model = {"kind": None, "obj": None}
def get_srgan(half):
    kind = "fp16" if half else "fp32"
    if _model["kind"] != kind:
        unload_srgan()
        _model["obj"], _model["kind"] = bench.FastSRGANUpscaler("Fast-SRGAN", "auto", 0, half), kind
    return _model["obj"]
def unload_srgan():
    if _model["obj"] is not None:
        _model["obj"], _model["kind"] = None, None
        gc.collect(); import torch; bench.gpu_free(torch, bench.pick_device())

def make_proc(kind, clip):
    W, H = clip["w"], clip["h"]
    if kind is None: return lambda f: f
    if kind in ("nearest", "bicubic", "lanczos"):
        unload_srgan()
        it = {"nearest": cv2.INTER_NEAREST, "bicubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}[kind]
        return lambda f: cv2.resize(f, (f.shape[1] * 4, f.shape[0] * 4), interpolation=it)
    if kind == "paste":
        unload_srgan()
        x0, y0, _, _ = bench.viewport_box(W, H, 120)
        def paste(f):
            full = np.zeros((H, W, 3), np.uint8); full[y0:y0 + f.shape[0], x0:x0 + f.shape[1]] = f; return full
        return paste
    sr = get_srgan(half=kind != "srgan32")
    vp = kind.endswith("_vp")
    lw, lh = even(W / 4), even(H / 4)
    region = (lw * lh * 2 / 9) if vp else lw * lh      # 120x120 deg = 1/3 of the width x 2/3 of the height
    sr.tile = 160 if region > 160_000 else 0           # big frames in 160x160 tiles so memory stays ~1 GB
    return bench.ViewportUpscaler(sr, 120, 4) if vp else sr

# ---------------------------------------------------------------- pipeline thread
state = {"method": "bicubic", "clip": "pano", "pending": True, "tile": 0}
frame_lock = threading.Condition(); latest = {"jpeg": None, "id": 0}
acc_lock = threading.Lock()
acc = {"frames": 0, "dec": 0.0, "proc": 0.0, "proc_max": 0.0, "jpeg": 0.0, "dropped": 0}
viewers = {"n": 0}   # open /live/stream connections; the pipeline idles when there are none
info = {"kbps": None, "stream_res": None, "fps": 25.0, "out_res": None}

def pipeline():
    cap = proc = None; n = 0; t0 = 0.0; fps = 25.0
    while True:
        if state["pending"]:
            m, cn = state["method"], state["clip"]
            c = clips.get(cn)
            if not c or c["status"] != "ready":
                time.sleep(0.2); continue
            state["pending"] = False
            stream, kind = TECH[m]
            src = f"prepared/{cn}/{stream}.mp4"
            try:
                proc = make_proc(kind, c)
            except Exception as e:
                print("failed to set up", m, e, flush=True); continue
            state["tile"] = getattr(_model["obj"], "tile", 0) if kind and kind.startswith("srgan") else 0
            if cap: cap.release()
            cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 150
            info.update(kbps=os.path.getsize(src) * 8 / 1000 / (nfr / fps), stream_res=f"{int(cap.get(3))}x{int(cap.get(4))}", fps=fps)
            n = 0; t0 = time.perf_counter()
        if cap is None: time.sleep(0.1); continue
        if viewers["n"] == 0:                      # nobody watching: don't burn CPU/GPU
            time.sleep(0.2); t0 = time.perf_counter() - n / fps; continue
        target = int((time.perf_counter() - t0) * fps)
        while n < target - 1:                      # a real player drops frames when it falls behind
            if not cap.grab(): break
            n += 1
            with acc_lock: acc["dropped"] += 1
        a = time.perf_counter(); ok, f = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0); n = 0; t0 = time.perf_counter(); continue
        b = time.perf_counter(); out = proc(f); c_ = time.perf_counter()
        jpeg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes(); d = time.perf_counter()
        info["out_res"] = f"{out.shape[1]}x{out.shape[0]}"
        n += 1
        with acc_lock:
            acc["frames"] += 1; acc["dec"] += b - a; acc["proc"] += c_ - b; acc["jpeg"] += d - c_
            acc["proc_max"] = max(acc["proc_max"], c_ - b)
        with frame_lock:
            latest["jpeg"] = jpeg; latest["id"] += 1; frame_lock.notify_all()
        wait = t0 + n / fps - time.perf_counter()
        if wait > 0: time.sleep(wait)

# ---------------------------------------------------------------- stats thread
history = collections.deque(maxlen=240)
def gpu_stats():
    try:
        s = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"], capture_output=True, text=True, timeout=2).stdout
        u = re.search(r'"Device Utilization %"=(\d+)', s); m = re.search(r'"In use system memory"=(\d+)', s)
        return (int(u.group(1)) if u else None, int(m.group(1)) / 2**20 if m else None)
    except Exception: return (None, None)
def rss_mb():
    try: return int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout) / 1024
    except Exception: return None
def stats():
    r0, t0 = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter()
    while True:
        time.sleep(0.5)
        r1, t1 = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter()
        cpu = ((r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)) / (t1 - t0) * 100
        r0, dt = r1, t1 - t0; t0 = t1
        with acc_lock:
            a = dict(acc); acc.update(frames=0, dec=0.0, proc=0.0, proc_max=0.0, jpeg=0.0, dropped=0)
        n = max(a["frames"], 1); util, gmem = gpu_stats(); mps = None
        if "torch" in sys.modules:
            try: mps = bench.gpu_mem_mb(sys.modules["torch"], bench.pick_device())
            except Exception: pass
        history.append({"t": time.time(), "method": state["method"], "clip": state["clip"],
            "fps": a["frames"] / dt, "dropped_per_s": a["dropped"] / dt,
            "proc_ms": 1000 * a["proc"] / n, "proc_max_ms": 1000 * a["proc_max"], "decode_ms": 1000 * a["dec"] / n,
            "jpeg_ms": 1000 * a["jpeg"] / n, "budget_ms": 1000 / info["fps"],
            "cpu_pct": cpu, "rss_mb": rss_mb(), "gpu_util_pct": util, "gpu_mem_mb": gmem, "mps_mb": mps,
            "kbps": info["kbps"], "stream_res": info["stream_res"], "out_res": info["out_res"],
            "tile": state["tile"], "torch_loaded": "torch" in sys.modules})

# ---------------------------------------------------------------- web server
class H(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
        if u.path == "/live/set":
            m, cn = q.get("m", [None])[0], q.get("clip", [None])[0]
            if m in TECH: state["method"] = m
            if cn in clips: state["clip"] = cn
            state["pending"] = True
            return self._json({"ok": True, "method": state["method"], "clip": state["clip"]})
        if u.path == "/live/clips":
            scan()
            return self._json({"current": state["clip"], "max_seconds": MAX_SECONDS,
                               "clips": [{"name": k, **{x: v for x, v in c.items()}} for k, c in clips.items()]})
        if u.path == "/live/stats":
            return self._json({"method": state["method"], "clip": state["clip"], "history": list(history)})
        if u.path == "/live/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store"); self.end_headers()
            last = -1; viewers["n"] += 1
            try:
                while True:
                    with frame_lock:
                        frame_lock.wait_for(lambda: latest["id"] != last, timeout=2)
                        j, last = latest["jpeg"], latest["id"]
                    if j is None: continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(j)).encode() + b"\r\n\r\n" + j + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                return
            finally:
                viewers["n"] -= 1
        return super().do_GET()
    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        if u.path != "/live/upload": return self.send_error(404)
        fn = os.path.basename(urllib.parse.unquote(self.headers.get("X-Filename", "upload.mp4")))
        base, ext = os.path.splitext(fn)
        if ext.lower() not in VIDEO_EXT: return self._json({"ok": False, "error": f"unsupported file type {ext}"})
        n = clip_name(fn); fn = n + ext.lower(); i = 2
        while os.path.exists(f"clips/{fn}") or n in clips:
            n = clip_name(base) + f"_{i}"; fn = n + ext.lower(); i += 1
        left = int(self.headers.get("Content-Length", 0))
        if left > 2 * 1024**3: return self._json({"ok": False, "error": "file over 2 GB"})
        with open(f"clips/{fn}", "wb") as fo:
            while left > 0:
                chunk = self.rfile.read(min(1 << 20, left))
                if not chunk: break
                fo.write(chunk); left -= len(chunk)
        scan()
        return self._json({"ok": True, "name": n})
    def _json(self, o):
        b = json.dumps(o).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(b))); self.end_headers()
        self.wfile.write(b)

if __name__ == "__main__":
    scan()
    for t in (prep_worker, pipeline, stats): threading.Thread(target=t, daemon=True).start()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), H); srv.daemon_threads = True
    print(f"Live viewer: http://localhost:{PORT}/viewer.html  (videos in clips/, max {MAX_SECONDS}s used)", flush=True)
    srv.serve_forever()
