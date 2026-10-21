"""
Accuracy evaluation of alerts against hand-labelled events.

ground truth JSON: [{"type": "PERIMETER_BREACH", "t": 20.1}, ...]   (t = video seconds)
  python scripts/evaluate.py --source samples/vtest.avi --truth samples/ground_truth.json --tol 2.0

Reports precision / recall / F1 per alert type and false alerts per minute.
An alert is a true positive if a not-yet-matched labelled event of the same type lies within +-tol seconds.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ibvap.config import load_config          # noqa: E402
from ibvap.engine import IBVAPEngine          # noqa: E402
from ibvap.pipeline import FrameSource        # noqa: E402


def score(pred, truth, tol):
    """pred/truth: lists of (type, t). Returns {type: {tp, fp, fn, precision, recall, f1}}."""
    out = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for typ in {t for t, _ in pred} | {t for t, _ in truth}:
        P = sorted(t for ty, t in pred if ty == typ)
        T = sorted(t for ty, t in truth if ty == typ)
        used = set()
        for tp_t in P:
            hit = next((i for i, g in enumerate(T) if i not in used and abs(g - tp_t) <= tol), None)
            if hit is None:
                out[typ]["fp"] += 1
            else:
                used.add(hit); out[typ]["tp"] += 1
        out[typ]["fn"] = len(T) - len(used)
    for r in out.values():
        r["precision"] = round(r["tp"] / (r["tp"] + r["fp"]), 3) if r["tp"] + r["fp"] else 0.0
        r["recall"] = round(r["tp"] / (r["tp"] + r["fn"]), 3) if r["tp"] + r["fn"] else 0.0
        pr = r["precision"] + r["recall"]
        r["f1"] = round(2 * r["precision"] * r["recall"] / pr, 3) if pr else 0.0
    return dict(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True); p.add_argument("--truth", required=True)
    p.add_argument("--config", default="config.yaml"); p.add_argument("--tol", type=float, default=2.0)
    p.add_argument("--backend"); p.add_argument("--model"); p.add_argument("--max-width", type=int, default=960)
    a = p.parse_args()
    cfg = load_config(a.config if os.path.exists(a.config) else None)
    if a.backend: cfg["model"]["backend"] = a.backend
    if a.model: cfg["model"]["name"] = a.model
    eng = IBVAPEngine(cfg)
    pred, last_ts = [], 0.0
    for frame, ts in FrameSource(a.source, max_width=a.max_width).frames():
        _, alerts = eng.process(frame, ts)
        last_ts = ts
        pred += [(x["type"], x["timestamp"]) for x in alerts if not x.get("suppressed")]
    truth = [(e["type"], float(e["t"])) for e in json.load(open(a.truth))]
    res = score(pred, truth, a.tol)
    print(f"{'type':<22}{'TP':>4}{'FP':>5}{'FN':>5}{'prec':>8}{'rec':>7}{'F1':>7}")
    for typ, r in sorted(res.items()):
        print(f"{typ:<22}{r['tp']:>4}{r['fp']:>5}{r['fn']:>5}{r['precision']:>8}{r['recall']:>7}{r['f1']:>7}")
    fp = sum(r["fp"] for r in res.values())
    print(f"\nfalse alerts per minute: {fp / max(last_ts / 60, 1e-6):.2f}  (video length {last_ts:.0f}s)")
    os.makedirs("out", exist_ok=True)
    json.dump(res, open("out/evaluation.json", "w"), indent=2)
