# IBVAP — AI-Based Intelligent Video Analytics Platform for Border Surveillance

**SIH26187 · Ministry of Home Affairs / Sashastra Seema Bal (SSB) · Smart India Hackathon 2026 · Software track**

IBVAP augments existing CCTV with real-time computer vision: it detects and tracks people and vehicles,
raises alerts for fence crossings, loitering, suspicious posture, abandoned bags, crowds and running,
reads number plates, matches faces against a watchlist, and gives the command centre a live dashboard
with evidence snapshots and PDF incident reports.

> AI detections are decision support. An operator must verify an alert before any action is taken.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run.py --source samples/vtest.avi --output out/annotated.mp4      # process a video
python run.py --source rtsp://user:pass@192.168.1.20/stream1 --display   # live camera (q to quit)
python run.py --serve                                                    # web dashboard -> http://127.0.0.1:8000
python -m unittest discover -s tests -t .                                # run the tests
```

The first run with YOLO downloads `yolov8n.pt` automatically. If `ultralytics` is not installed (or offline),
IBVAP falls back to a lightweight OpenCV **motion backend** (demo/testing only, much less accurate:
it cannot tell a crouching person from two people standing together).

## Architecture

```
 camera / file / RTSP ──► FrameSource (auto-reconnect, video-time clock)
                               │
                  one IBVAPEngine per camera (own tracker + state)
                               ▼
 enhance (CLAHE night / thermal) ─► YOLO + ByteTrack (or pose model) ─► tracks
                               │
     ┌──────────────┬──────────┴─────────┬──────────────┬──────────────┐
  Tripwire &      Behaviour           ANPR           Face watch      Threat level
  zones (dir,    (posture, bags,   (plate find → OCR   (YuNet+SFace    (hold timer)
  dwell)          crowd, running)   → IN format fix     embeddings)
                                      → voting)
     └──────────────┴──────────┬─────────┴──────────────┴──────────────┘
                               ▼
      AlertStore (SQLite + JPEG evidence + dedupe) ─► webhook / Telegram
                               ▼
      Flask dashboard: live MJPEG wall · alert feed · acknowledge · PDF incident / shift reports
```

## Project structure

```
run.py                  entry point            config.yaml   all settings (tripwires, zones, cameras...)
ibvap/
  config.py             defaults + YAML merge  detector.py   YOLO+ByteTrack (+pose) / motion fallback
  engine.py             per-camera pipeline    tripwire.py   crossing direction, zones, loitering
  behavior.py           posture/bag/crowd/run  anpr.py        plate OCR + Indian format correction
  face_watch.py         face embeddings        enhancer.py   night / thermal
  alerts.py             SQLite store + notify  report.py     PDF reports
  pipeline.py           sources + threads      dashboard.py  web UI + REST API
scripts/                benchmark.py · evaluate.py · enroll_face.py · download_models.py
tests/                  48 unit tests
```

## Configuration highlights (`config.yaml`)

| Want | Setting |
|---|---|
| Better accuracy for small distant people | `model.name: yolov8s.pt` or `yolo11s.pt`, `model.imgsz: 960` |
| Crawl/crouch from skeleton, not box shape | `model.pose_model: yolov8n-pose.pt` |
| Draw your fence | `tripwires[].p1/p2` (0–1 fractions of the frame), `direction: in/out/both` |
| Restricted area + dwell limit | `zones[].polygon`, `max_dwell_sec` |
| More cameras | add entries under `cameras:` (file, webcam index or `rtsp://`) — each gets its own engine/thread |
| Phone/SMS alerts | `alerts.webhook_url` or `telegram_bot_token` + `telegram_chat_id` |
| Real plate watchlist | JSON `{"DL01AB1234": "reason"}` at `anpr.watchlist_file` (the built-in list is synthetic demo data) |

Tripwire direction: for wire `p1→p2`, `side = (x2-x1)(py-y1) − (y2-y1)(px-x1)`. **"in"** = crossing from side<0 to side>0
(for a horizontal wire: top → bottom).

## Face watchlist (optional)

```bash
python scripts/download_models.py        # YuNet + SFace, ~40 MB from the OpenCV model zoo
python scripts/enroll_face.py POI_001 --name "Name" --category "Category" --risk HIGH photo1.jpg photo2.jpg
# then set face.enabled: true in config.yaml
```
Manual download if the script is blocked: `face_detection_yunet_2023mar.onnx` and `face_recognition_sface_2021dec.onnx`
from <https://github.com/opencv/opencv_zoo> (folders `models/face_detection_yunet`, `models/face_recognition_sface`) into `models/`.
Use only lawfully held photos; real deployments need legal authority, access control and audit (every match is logged in the alert DB).

## Measuring your results (for the SIH presentation)

```bash
python scripts/benchmark.py --backend yolo --frames 300     # FPS and per-stage latency -> out/benchmark.json
python scripts/evaluate.py --source samples/vtest.avi --truth samples/ground_truth.json
#   label events by hand: [{"type": "PERIMETER_BREACH", "t": 20.1}, ...] -> precision / recall / F1, false alerts per minute
```

## REST API

`GET /api/cameras` · `GET /api/alerts?limit&camera&min_severity` · `GET /api/stats` · `POST /api/alerts/<id>/ack` ·
`GET /api/alerts/<id>/report.pdf` · `GET /api/report/summary.pdf` · `POST /api/cameras/<id>/mode {"night":true}` · `GET /video/<cam>` (MJPEG)

The dashboard has no login. Bind it to `127.0.0.1` (default) or put it behind a reverse proxy with authentication before exposing it on a network.

## What changed from v1

Fixed: loitering could never fire (5 s history vs 10 s dwell) · direction-aware crossings were not implemented ·
cameras shared one tracker/state · breach status flickered for a single frame · emoji drawn by OpenCV rendered as `???` ·
bags/bicycles raised CRITICAL breaches · timers used wall clock (wrong for video files) · memory leaks in the cooldown cache ·
pixel thresholds ignored resolution · posture alerts fired on single noisy frames.
Added: config file, CLI, real OCR ANPR with voting, face embeddings, crowd + running detection, alert database,
snapshots, PDF reports, webhook/Telegram, multi-camera dashboard, tests, benchmark and evaluation scripts.

## License

MIT — built for Smart India Hackathon 2026, problem statement SIH26187.
