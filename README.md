<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:0a2540,100:1a6cf7&height=220&section=header&text=IBVAP&fontSize=90&fontColor=ffffff&fontAlignY=38&desc=Intelligent%20Border%20Video%20Analytics%20Platform&descAlignY=62&descColor=7dd3fc&animation=fadeIn" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![YOLO](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6600?style=for-the-badge&logo=yolo&logoColor=white)](https://ultralytics.com)
[![Flask](https://img.shields.io/badge/Flask-Dashboard-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![SIH](https://img.shields.io/badge/SIH2026-SIH26187-f59e0b?style=for-the-badge&logo=government&logoColor=white)](https://sih.gov.in)
[![MHA](https://img.shields.io/badge/Ministry-Home%20Affairs%20%2F%20SSB-dc2626?style=for-the-badge)](https://mha.gov.in)

<br/>

> **AI-augmented real-time perimeter security for India's borders.**  
> Detects intrusions, reads number plates, matches faces, raises alerts — all from existing CCTV infrastructure.

<br/>

[🚀 Quick Start](#-quick-start) · [🧠 Architecture](#-architecture) · [⚙️ Configuration](#️-configuration) · [🖥️ Dashboard](#️-command-center-dashboard) · [📡 REST API](#-rest-api) · [🧪 Testing](#-testing) · [📊 Benchmarking](#-benchmarking)

</div>

---

## 🎯 What is IBVAP?

**IBVAP** (**I**ntelligent **B**order **V**ideo **A**nalytics **P**latform) is an AI-powered video analytics system built for the **Smart India Hackathon 2026** under problem statement **SIH26187** for the **Ministry of Home Affairs / Sashastra Seema Bal (SSB)**.

It transforms existing CCTV infrastructure into an intelligent, real-time threat detection network — without replacing any hardware.

<div align="center">

| 🔍 Detect & Track | 🚨 Alert & Notify | 🖥️ Monitor |
|:---:|:---:|:---:|
| People, vehicles, wildlife | SQLite + Telegram + Webhook | Live MJPEG multi-cam wall |
| Crawling / crouching infiltrators | CRITICAL / HIGH / MEDIUM / LOW | Per-alert PDF incident reports |
| Abandoned baggage, crowds, running | Operator acknowledge workflow | Shift summary PDF exports |
| Indian number plates (ANPR) | Evidence JPEG snapshots | Night-vision & thermal views |
| Watchlist face matching | Auto-deduplication | REST API for integrations |

</div>

> ⚠️ **Operator-in-the-loop by design.** AI detections are decision support tools. An operator must verify every alert before any action is taken.

---

## 🧠 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        INPUT SOURCES                                    │
│         📁 Video File  │  📷 Webcam  │  📡 RTSP / HTTP Stream          │
└──────────────────────────────┬──────────────────────────────────────────┘
                               ▼
                    ┌──────────────────┐
                    │   FrameSource    │  auto-reconnect · video-clock sync
                    └────────┬─────────┘
                             │  one CameraWorker thread per camera
                             ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        IBVAPEngine (per camera)                        │
│                                                                        │
│  ① Enhance ──► CLAHE night-vision  /  Inferno thermal pseudo-colour   │
│                                                                        │
│  ② Detect ───► YOLOv8 + ByteTrack  (fallback: MOG2 motion detector)  │
│                + optional YOLOv8-Pose for keypoint-based posture       │
│                                                                        │
│  ③ Analyse ──► ┌──────────────┬────────────┬───────────┬───────────┐  │
│               │  Tripwire &  │ Behaviour  │   ANPR    │   Face    │  │
│               │    Zones     │            │           │ Watchlist │  │
│               │              │ • Crawl    │ • Locate  │           │  │
│               │ • Crossing   │ • Crouch   │ • OCR     │ • YuNet   │  │
│               │   direction  │ • Running  │ • Format  │   detect  │  │
│               │ • Dwell /    │ • Crowd    │   correct │ • SFace   │  │
│               │   loitering  │ • Bags     │ • Vote    │   embed   │  │
│               └──────────────┴────────────┴───────────┴───────────┘  │
│                                                                        │
│  ④ Threat ───► SECURE → ELEVATED → CRITICAL  (5s hold timer)         │
│                                                                        │
│  ⑤ HUD ──────► Tactical overlay: boxes · banners · wires · zones     │
└──────────────────────────────┬─────────────────────────────────────────┘
                               ▼
            ┌──────────────────────────────────┐
            │         AlertStore               │
            │  SQLite · JPEG snapshots         │
            │  Dedup · Webhook · Telegram      │
            └──────────────────┬───────────────┘
                               ▼
     ┌─────────────────────────────────────────────────┐
     │          Flask Command-Center Dashboard          │
     │  Live MJPEG · Alert feed · ACK · PDF reports    │
     │  REST API · Night/Thermal toggle per camera      │
     └─────────────────────────────────────────────────┘
```

---

## ✨ Feature Highlights

<details>
<summary><b>🛡️ Perimeter Intrusion Detection</b></summary>

- **Direction-aware virtual tripwires** — configurable `in` / `out` / `both` crossing detection
- Uses the object's **foot point** on the wire segment (not bounding-box center) for precision
- Per-class filtering — only relevant classes (person, bicycle, motorcycle, car, bus, truck) raise `PERIMETER_BREACH`
- Animals raise a suppressed `WILDLIFE_CROSSING` (logged but not notified)
- **Exclusion zones** (polygon geofences) with configurable dwell-time limits and optional entry alerts
- Cooldown cache and stale-track pruning — no memory leaks, no duplicate alerts

</details>

<details>
<summary><b>🕵️ Behavioural Anomaly Detection</b></summary>

- **Crawling / Crouching** — temporal EMA of bounding-box aspect ratio, confirmed over N frames to eliminate false alarms; optional COCO-17 keypoint posture from a pose model
- **Rapid Movement (Running)** — EMA speed in body-heights per second, sustained over N frames
- **Crowd Formation** — union-find connected-components by proximity, fires after a configurable hold window
- **Unattended Baggage** — tracks `backpack / suitcase / handbag` with no nearby person for N seconds

</details>

<details>
<summary><b>🚗 ANPR — Indian Number Plate Recognition</b></summary>

- Supports **Civilian** (`DL 01 AB 1234`), **Bharat Series** (`22 BH 1234 AA`) and **Military** formats
- Morphological plate localisation on the lower vehicle crop — no separate plate-detector model needed
- Positional OCR-error correction: `O↔0`, `I↔1`, `Z↔2`, `S↔5`, `B↔8` etc. by character position
- Per-track **majority-vote confirmation** (default 3 votes) eliminates OCR flicker
- Pluggable backends: **EasyOCR** (preferred) → **Tesseract** → no-OCR fallback (plate locate only)
- JSON watchlist — flag any plate for immediate `WATCHLIST_VEHICLE` CRITICAL alert

</details>

<details>
<summary><b>👤 Watchlist Face Recognition</b></summary>

- **YuNet** ONNX face detector + **SFace** ONNX 128-d embedding — both run via OpenCV, no extra pip packages
- Cosine-similarity matching with configurable threshold (default 0.363)
- Per-track cooldown (30 s) prevents alert flooding for the same person
- Enrolment CLI: `python scripts/enroll_face.py POI_001 --name "..." --category "..." --risk HIGH photo.jpg`
- Gracefully disabled if model files are not downloaded (zero-crash startup)

</details>

<details>
<summary><b>🌙 Night-Vision & Thermal Enhancement</b></summary>

- **CLAHE** in LAB colour space (luminance only — colour is preserved)
- Gamma correction LUT for additional brightness boost
- **Pseudo-thermal** rendering: grayscale CLAHE → OpenCV `COLORMAP_INFERNO`
- Auto-mode: applies enhancement only when mean frame luminance drops below threshold
- Toggle per-camera at runtime from the dashboard (`Night` / `Thermal` buttons)

</details>

---

## 📁 Project Structure

```
IBVAP/
├── run.py                        # Entry point
├── config.yaml                   # All settings (edit this)
├── requirements.txt
│
├── ibvap/
│   ├── config.py                 # DEFAULTS + YAML deep-merge
│   ├── engine.py                 # Per-camera analytics pipeline
│   ├── detector.py               # YOLO+ByteTrack / MotionDetector
│   ├── behavior.py               # Posture / running / crowd / bags
│   ├── tripwire.py               # Crossing direction + zone loitering
│   ├── anpr.py                   # Indian plate OCR + voting
│   ├── face_watch.py             # YuNet + SFace watchlist matching
│   ├── alerts.py                 # SQLite store + notifications
│   ├── dashboard.py              # Flask web UI + REST API
│   ├── report.py                 # PDF incident & shift reports
│   ├── pipeline.py               # FrameSource + CameraWorker threads
│   ├── enhancer.py               # Night-vision + thermal
│   └── cli.py                    # CLI argument parser
│
├── scripts/
│   ├── benchmark.py              # FPS + per-stage latency → out/benchmark.json
│   ├── evaluate.py               # Precision / Recall / F1 vs ground truth
│   ├── enroll_face.py            # Enrol watchlist face photos
│   └── download_models.py        # Download YuNet + SFace ONNX (~40 MB)
│
├── tests/                        # 48 unit tests
├── samples/
│   ├── vtest.avi                 # Bundled test video
│   └── ground_truth_example.json
├── data/                         # Runtime (DB, snapshots, reports) — gitignored
└── models/                       # ONNX face models — gitignored
```

---

## 🚀 Quick Start

### 1 — Install

```bash
# Clone the repo
git clone https://github.com/GOWTH4M-444/IBVAP.git
cd IBVAP

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux / macOS)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> The first YOLO run auto-downloads `yolov8n.pt` (~6 MB). If ultralytics is unavailable, IBVAP falls back to a lightweight OpenCV **motion backend** (demo/testing only).

### 2 — Run

```bash
# Process a video file → annotated MP4
python run.py --source samples/vtest.avi --output out/annotated.mp4

# Live camera stream with display window (press q to quit)
python run.py --source rtsp://user:pass@192.168.1.20/stream1 --display

# Launch the web command-center dashboard
python run.py --serve
# → open http://127.0.0.1:8000

# Force night-vision or thermal mode
python run.py --source samples/vtest.avi --night
python run.py --source samples/vtest.avi --thermal

# Use a lighter / heavier YOLO model
python run.py --source samples/vtest.avi --model yolov8s.pt
```

### 3 — Face Recognition Setup (Optional)

```bash
# Download YuNet + SFace ONNX models (~40 MB)
python scripts/download_models.py

# Enrol a person of interest
python scripts/enroll_face.py POI_001 --name "John Doe" --category "Smuggler" --risk HIGH photo1.jpg photo2.jpg

# Enable face matching in config.yaml
# face:
#   enabled: true
```

---

## ⚙️ Configuration

All settings live in [`config.yaml`](config.yaml). Everything not specified falls back to defaults in `ibvap/config.py` — partial configs are always valid.

| Want | Setting |
|---|---|
| Better accuracy for small/distant people | `model.name: yolov8s.pt`, `model.imgsz: 960` |
| Keypoint-based crawl/crouch | `model.pose_model: yolov8n-pose.pt` |
| Custom fence line | `tripwires[].p1/p2` in 0–1 fractions of frame |
| Crossing direction | `tripwires[].direction: in / out / both` |
| Restricted area + dwell limit | `zones[].polygon`, `zones[].max_dwell_sec` |
| Multiple cameras | Add entries under `cameras:` (file / webcam index / `rtsp://`) |
| Telegram alerts | `alerts.telegram_bot_token` + `alerts.telegram_chat_id` |
| Webhook alerts | `alerts.webhook_url` (receives JSON POST) |
| Real plate watchlist | `anpr.watchlist_file: data/plate_watchlist.json` |
| Night-vision auto-threshold | `enhancer.low_light_threshold: 70.0` |
| Alert severity minimum for notify | `alerts.notify_min_severity: HIGH` |

### Tripwire Direction Math

For a wire from `p1 → p2`:
```
side(pt) = (x2 - x1) × (py - y1) − (y2 - y1) × (px - x1)
```
- `"in"`  = object crosses from `side < 0` to `side > 0`  *(for a horizontal wire: top → bottom)*
- `"out"` = opposite direction
- `"both"` = either direction raises an alert

---

## 🖥️ Command-Center Dashboard

Launch with `python run.py --serve` → [http://127.0.0.1:8000](http://127.0.0.1:8000)

```
┌─────────────────────────────────────────────────────────────────┐
│  IBVAP COMMAND CENTER           ● SECURE          SHIFT REPORT  │
├────────────────────────────┬────────────────────────────────────┤
│                            │  ALERTS                            │
│  ┌──────────┐ ┌──────────┐ │  ─────────────────────────────── │
│  │  CAM_01  │ │  CAM_02  │ │  [CRITICAL] PERIMETER BREACH...  │
│  │  MJPEG   │ │  MJPEG   │ │  [HIGH]     SUSPICIOUS POSTURE.. │
│  │  stream  │ │  stream  │ │  [HIGH]     WATCHLIST MATCH...   │
│  └──────────┘ └──────────┘ │  [MEDIUM]   RAPID MOVEMENT...    │
│  Night  Thermal             │                                   │
│                            │  [snapshot] [report] [acknowledge] │
└────────────────────────────┴────────────────────────────────────┘
```

**Features:**
- 🎥 Live MJPEG streams for every camera (auto-updates every frame)
- 🚨 Real-time alert feed with colour-coded severity (CRITICAL / HIGH / MEDIUM)
- 📸 Expandable JPEG evidence snapshots per alert
- ✅ Operator acknowledge workflow with free-text notes
- 📄 Per-alert PDF incident reports (click **report** on any alert)
- 📋 Shift summary PDF (`Shift report` link in the header)
- 🌙 Night-vision / Thermal toggle per camera
- 🔴 Flashing red border overlay during active CRITICAL breach

---

## 📡 REST API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/cameras` | Status of all cameras (FPS, threat level, persons, backend) |
| `GET` | `/api/alerts` | Alert list — query: `limit`, `camera`, `min_severity` |
| `GET` | `/api/stats` | Total alerts grouped by severity and type |
| `POST` | `/api/alerts/<id>/ack` | Acknowledge an alert `{"note": "..."}` |
| `GET` | `/api/alerts/<id>/report.pdf` | Download PDF incident report |
| `GET` | `/api/report/summary.pdf` | Download shift summary PDF |
| `POST` | `/api/cameras/<id>/mode` | Toggle night/thermal `{"night": true}` |
| `GET` | `/video/<cam>` | MJPEG stream for a camera |
| `GET` | `/snapshots/<name>` | Serve evidence JPEG |

---

## 🧪 Testing

```bash
# Run all 48 unit tests
python -m unittest discover -s tests -t .

# Run a specific module
python -m unittest tests.test_tripwire
python -m unittest tests.test_anpr
python -m unittest tests.test_behavior
python -m unittest tests.test_engine
python -m unittest tests.test_alerts_report
python -m unittest tests.test_config_dashboard
```

| Test File | What it covers |
|---|---|
| `test_engine.py` | Full pipeline, threat levels, ANPR/face enrichment |
| `test_tripwire.py` | Crossings, direction filtering, zone dwell, cooldowns |
| `test_behavior.py` | Posture, running speed, crowd, abandoned baggage |
| `test_anpr.py` | Plate parsing, positional correction, voting, watchlist |
| `test_alerts_report.py` | SQLite store, dedup, subscriber queue, PDF generation |
| `test_config_dashboard.py` | Config deep merge, Flask routes |
| `test_scripts.py` | Script imports and CLI invocability |

---

## 📊 Benchmarking & Evaluation

### Benchmark (FPS + latency per stage)

```bash
python scripts/benchmark.py --source samples/vtest.avi --frames 300
# Writes → out/benchmark.json

# With a specific backend
python scripts/benchmark.py --backend yolo --model yolov8s.pt --frames 300
```

**Output (`out/benchmark.json`):**
```json
{
  "backend": "yolo",
  "model": "yolov8n.pt",
  "fps": 28.4,
  "latency_ms_mean": 35.2,
  "latency_ms_p95": 48.1,
  "stage_ms_mean": {
    "enhance": 0.3,
    "detect": 28.1,
    "behavior": 1.2,
    "tripwire": 0.5,
    "anpr": 3.8,
    "face": 0.2,
    "hud": 1.1
  }
}
```

### Accuracy Evaluation (Precision / Recall / F1)

```bash
# Create your ground-truth file: [{"type": "PERIMETER_BREACH", "t": 20.1}, ...]
python scripts/evaluate.py --source samples/vtest.avi --truth samples/ground_truth.json
# Writes → out/evaluation.json
```

```
type                    TP   FP   FN    prec     rec     F1
PERIMETER_BREACH         8    1    2   0.889   0.800   0.842
SUSPICIOUS_POSTURE       3    0    1   1.000   0.750   0.857

false alerts per minute: 0.12  (video length 312s)
```

---

## 🗺️ Roadmap

- [x] YOLO v8 / v11 detection + ByteTrack
- [x] Direction-aware virtual tripwires
- [x] Exclusion zones + dwell loitering
- [x] Crawl / crouch / run / crowd / bag detection
- [x] Indian ANPR with majority-vote OCR
- [x] Face watchlist (YuNet + SFace)
- [x] Flask multi-camera command-center dashboard
- [x] PDF incident + shift summary reports
- [x] Webhook + Telegram alert notifications
- [x] Night-vision CLAHE + pseudo-thermal
- [x] 48-test unit test suite
- [ ] Drone / PTZ camera integration
- [ ] Heatmap of intrusion hotspots
- [ ] Event timeline replay
- [ ] Login / role-based access for the dashboard
- [ ] Multi-site (distributed) deployment mode

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feat/your-feature`
3. Commit your changes: `git commit -m 'feat: describe your change'`
4. Push to the branch: `git push origin feat/your-feature`
5. Open a Pull Request

Please ensure all tests pass (`python -m unittest discover -s tests -t .`) before submitting a PR.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🏛️ Acknowledgements

- **Smart India Hackathon 2026** — Problem Statement SIH26187
- **Ministry of Home Affairs / Sashastra Seema Bal (SSB)**
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — object detection + ByteTrack
- [OpenCV Zoo](https://github.com/opencv/opencv_zoo) — YuNet & SFace ONNX models
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) — ANPR optical character recognition
- [ReportLab](https://www.reportlab.com/) — PDF incident reports
- [Flask](https://flask.palletsprojects.com/) — command-center web dashboard

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a6cf7,50:0a2540,100:0d1117&height=120&section=footer" width="100%"/>

**Built with ❤️ for Smart India Hackathon 2026**  
*Securing India's Borders with AI*

[![GitHub](https://img.shields.io/badge/GitHub-GOWTH4M--444%2FIBVAP-181717?style=for-the-badge&logo=github)](https://github.com/GOWTH4M-444/IBVAP)

</div>
