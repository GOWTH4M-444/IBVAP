"""
IBVAP command line.

  python run.py --source samples/vtest.avi --output out/annotated.mp4
  python run.py --source rtsp://user:pass@host/stream --display
  python run.py --serve                     # dashboard for every camera in config.yaml
"""

import argparse
import os
import sys
import time

import cv2

from .alerts import AlertStore
from .config import load_config
from .engine import IBVAPEngine
from .pipeline import FrameSource


def run_headless(args, cfg) -> int:
    store = AlertStore(cfg["alerts"]) if not args.no_store else None
    engine = IBVAPEngine(cfg, args.camera_id, args.sector, alert_store=store)
    src = FrameSource(args.source, max_width=args.max_width)
    writer, n, t0, counts = None, 0, time.time(), {}
    try:
        for frame, ts in src.frames():
            out, alerts = engine.process(frame, ts, args.night, args.thermal)
            n += 1
            for a in alerts:
                counts[a["type"]] = counts.get(a["type"], 0) + 1
                if not a.get("suppressed"):
                    print(f"[{a['severity']:<8}] t={ts:7.2f}s {a['camera_id']} {a['message']}")
            if args.output:
                if writer is None:
                    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
                    h, w = out.shape[:2]
                    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), args.out_fps, (w, h))
                writer.write(out)
            if args.display:
                cv2.imshow("IBVAP", out)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if args.max_frames and n >= args.max_frames:
                break
    finally:
        if writer:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()
    dt = max(1e-6, time.time() - t0)
    print(f"\n[IBVAP] {n} frames in {dt:.1f}s ({n / dt:.1f} fps) | backend={engine.detector.name} | alerts={counts or 'none'}")
    if args.output:
        print(f"[IBVAP] annotated video -> {args.output}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ibvap", description="IBVAP border video analytics")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--source", help="video file, webcam index or rtsp/http URL")
    p.add_argument("--camera-id", default="CAM_01")
    p.add_argument("--sector", default="Sector 4 (Fence Perimeter)")
    p.add_argument("--output", help="write annotated mp4 here")
    p.add_argument("--out-fps", type=float, default=15.0)
    p.add_argument("--display", action="store_true", help="show a live window (press q to quit)")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--max-width", type=int, default=960, help="downscale wide frames (0 = off)")
    p.add_argument("--night", action="store_true", help="force night enhancement")
    p.add_argument("--thermal", action="store_true", help="thermal view")
    p.add_argument("--backend", choices=["auto", "yolo", "motion"], help="override model.backend")
    p.add_argument("--model", help="override model.name, e.g. yolo11n.pt")
    p.add_argument("--no-store", action="store_true", help="do not write alerts/snapshots to the database")
    p.add_argument("--serve", action="store_true", help="start the web dashboard (all cameras in config)")
    p.add_argument("--host"); p.add_argument("--port", type=int)
    args = p.parse_args(argv)

    cfg = load_config(args.config if os.path.exists(args.config) else None)
    if args.backend:
        cfg["model"]["backend"] = args.backend
    if args.model:
        cfg["model"]["name"] = args.model
    if args.serve:
        from .dashboard import serve
        return serve(cfg, args.host, args.port)
    if not args.source:
        args.source = cfg["cameras"][0]["source"]
        print(f"[IBVAP] no --source given, using {args.source}")
    return run_headless(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
