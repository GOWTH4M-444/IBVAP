"""
Speed benchmark: per-stage latency and end-to-end FPS on a video.

  python scripts/benchmark.py --source samples/vtest.avi --frames 300 [--backend yolo --model yolov8n.pt]
Writes out/benchmark.json (paste the table into your SIH presentation).
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ibvap.config import load_config          # noqa: E402
from ibvap.engine import IBVAPEngine          # noqa: E402
from ibvap.pipeline import FrameSource        # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--source", default="samples/vtest.avi")
p.add_argument("--config", default="config.yaml")
p.add_argument("--frames", type=int, default=300)
p.add_argument("--backend"); p.add_argument("--model")
p.add_argument("--max-width", type=int, default=960)
a = p.parse_args()

cfg = load_config(a.config if os.path.exists(a.config) else None)
if a.backend: cfg["model"]["backend"] = a.backend
if a.model: cfg["model"]["name"] = a.model
eng = IBVAPEngine(cfg)
stages, total, n = {}, [], 0
for frame, ts in FrameSource(a.source, max_width=a.max_width).frames():
    t0 = time.perf_counter()
    eng.process(frame, ts)
    total.append((time.perf_counter() - t0) * 1000)
    for k, v in eng.last_stage_ms.items():
        stages.setdefault(k, []).append(v)
    n += 1
    if n >= a.frames:
        break
warm = total[10:] or total
res = {"backend": eng.detector.name, "model": cfg["model"]["name"], "frames": n,
       "fps": round(1000 / statistics.mean(warm), 1), "latency_ms_mean": round(statistics.mean(warm), 1),
       "latency_ms_p95": round(sorted(warm)[int(0.95 * (len(warm) - 1))], 1),
       "stage_ms_mean": {k: round(statistics.mean(v[10:] or v), 2) for k, v in stages.items()}}
os.makedirs("out", exist_ok=True)
json.dump(res, open("out/benchmark.json", "w"), indent=2)
print(json.dumps(res, indent=2))
