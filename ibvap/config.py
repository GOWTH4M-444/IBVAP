"""
Central configuration.

All tunables live in ``config.yaml`` (see repo root). Anything missing from the
YAML falls back to DEFAULTS below, so a partial config file is always valid.
"""

import copy
import os
from typing import Any, Dict, Optional

import yaml

DEFAULTS: Dict[str, Any] = {
    "model": {
        # backend: "auto" (YOLO if installed, else motion demo), "yolo", "motion"
        "backend": "auto",
        "name": "yolov8n.pt",        # try yolo11n.pt / yolov8s.pt for accuracy
        "conf": 0.35,
        "imgsz": 640,                # raise to 960/1280 for small, distant people
        "device": "auto",            # auto | cpu | cuda | mps
        "tracker": "bytetrack.yaml",
        "pose_model": None,          # e.g. "yolov8n-pose.pt" to enable keypoint posture
    },
    "enhancer": {
        "auto_low_light": True,
        "low_light_threshold": 70.0,
        "clahe_clip": 3.0,
    },
    # Normalized (0-1) coordinates. direction: both | in | out
    #   side(pt) = (x2-x1)*(py-y1) - (y2-y1)*(px-x1)
    #   "in"  = crossing from side<0 to side>0 (for the default horizontal wire: top -> bottom)
    "tripwires": [
        {
            "id": "border_fence_alpha",
            "name": "Border Fence - Sector 4",
            "p1": [0.1, 0.65],
            "p2": [0.9, 0.65],
            "direction": "both",
            "classes": ["person", "bicycle", "motorcycle", "car", "bus", "truck"],
        }
    ],
    "zones": [
        {
            "id": "restricted_bunker_zone",
            "name": "Buffer Zone 14",
            "polygon": [[0.2, 0.7], [0.8, 0.7], [0.85, 0.95], [0.15, 0.95]],
            "max_dwell_sec": 10.0,
            "alert_on_entry": False,
        }
    ],
    "tripwire": {
        "cross_cooldown_sec": 10.0,
        "loiter_cooldown_sec": 15.0,
        "track_stale_sec": 2.0,     # keep per-track state this long after it disappears
        "zone_exit_grace_sec": 1.5,  # ignore brief detection flicker when leaving a zone
    },
    "behavior": {
        "pose_method": "auto",       # auto (keypoints if present, else aspect) | aspect
        "crouch_aspect": 0.75,
        "crawl_aspect": 1.25,
        "pose_confirm_frames": 6,    # posture must persist this many observations
        "min_person_height_frac": 0.04,
        "edge_margin_frac": 0.01,    # skip boxes touching the frame edge (truncated)
        "pose_cooldown_sec": 12.0,
        "bag_type": ["backpack", "suitcase", "handbag"],
        "bag_stationary_sec": 15.0,
        "bag_move_frac": 0.02,       # fraction of frame diagonal
        "bag_person_near_frac": 0.12,
        "bag_cooldown_sec": 20.0,
        "crowd_min_people": 5,
        "crowd_radius_frac": 0.12,
        "crowd_hold_sec": 3.0,
        "crowd_cooldown_sec": 30.0,
        "run_speed_body_heights_per_sec": 2.5,
        "run_confirm_frames": 4,
        "run_cooldown_sec": 15.0,
    },
    "anpr": {
        "enabled": True,
        "ocr_backend": "auto",       # auto | easyocr | tesseract | none
        "every_n_frames": 5,
        "min_votes": 3,
        "min_vehicle_px": 80,
        "watchlist_file": "data/plate_watchlist.json",
    },
    "face": {
        "enabled": False,            # turn on after enrolling a watchlist
        "every_n_frames": 6,
        "backend": "auto",           # auto | sface | insightface
        "match_threshold": 0.363,    # SFace cosine threshold (higher = stricter)
        "watchlist_dir": "data/watchlist",
        "detector_model": "models/face_detection_yunet_2023mar.onnx",
        "recognizer_model": "models/face_recognition_sface_2021dec.onnx",
    },
    "alerts": {
        "db_path": "data/ibvap.db",
        "snapshot_dir": "data/snapshots",
        "report_dir": "data/reports",
        "dedupe_sec": 2.0,
        "webhook_url": None,         # POST JSON for HIGH/CRITICAL alerts
        "telegram_bot_token": None,
        "telegram_chat_id": None,
        "notify_min_severity": "HIGH",
    },
    "cameras": [
        {"id": "CAM_01", "sector": "Sector 4 (Fence Perimeter)", "source": "samples/vtest.avi"}
    ],
    "server": {"host": "127.0.0.1", "port": 8000, "jpeg_quality": 70},
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:  # lists (tripwires, zones, cameras) are replaced, not merged
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load config.yaml (if present) on top of DEFAULTS."""
    user: Dict[str, Any] = {}
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
    elif path:
        raise FileNotFoundError(f"Config file not found: {path}")
    return _deep_merge(DEFAULTS, user)
