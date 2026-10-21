"""
Pluggable detection + tracking backends.

  * YoloDetector   - Ultralytics YOLO + ByteTrack (the production backend).
                     Optional pose model gives keypoints for posture analysis.
  * MotionDetector - OpenCV background-subtraction + centroid tracker. Needs no model
                     weights or torch; used for offline demos, CI and unit tests.

Both return the same track dicts:
  {"id", "class", "conf", "bbox", "center", "center_norm", "foot", "foot_norm", ["kpts"]}
"""

import math
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# Tactical surveillance classes retained from COCO weights
TARGET_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    24: "backpack", 26: "handbag", 28: "suitcase",
}


def make_track(tid: int, cls: str, conf: float, box, w: int, h: int, kpts=None) -> Dict:
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    fx, fy = cx, y2
    t = {
        "id": tid, "class": cls, "conf": round(conf, 2),
        "bbox": (round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)),
        "center": (round(cx, 1), round(cy, 1)),
        "center_norm": (round(cx / w, 4), round(cy / h, 4)),
        "foot": (round(fx, 1), round(fy, 1)),
        "foot_norm": (round(fx / w, 4), round(fy / h, 4)),
    }
    if kpts is not None:
        t["kpts"] = kpts
    return t


def resolve_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# ----------------------------------------------------------------------------- YOLO
class YoloDetector:
    name = "yolo"

    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.35, imgsz: int = 640,
                 device: str = "auto", tracker: str = "bytetrack.yaml", pose_model: Optional[str] = None):
        from ultralytics import YOLO  # lazy: torch is heavy
        self.conf, self.imgsz, self.tracker = conf, imgsz, tracker
        self.device = resolve_device(device)
        print(f"[IBVAP] Loading YOLO '{model_name}' on {self.device} (imgsz={imgsz})")
        self.model = YOLO(model_name)
        self.pose = YOLO(pose_model) if pose_model else None
        self.model(np.zeros((320, 320, 3), np.uint8), verbose=False, device=self.device)  # warm-up

    def process_frame(self, frame_bgr: np.ndarray) -> List[Dict]:
        h, w = frame_bgr.shape[:2]
        res = self.model.track(frame_bgr, persist=True, conf=self.conf, imgsz=self.imgsz,
                               classes=list(TARGET_CLASSES), tracker=self.tracker,
                               verbose=False, device=self.device)
        tracks: List[Dict] = []
        if not res or res[0].boxes is None or len(res[0].boxes) == 0:
            return tracks
        for box in res[0].boxes:
            cls_id = int(box.cls[0].item())
            tid = int(box.id[0].item()) if box.id is not None else -1
            tracks.append(make_track(tid, TARGET_CLASSES.get(cls_id, "unknown"), float(box.conf[0].item()),
                                     box.xyxy[0].tolist(), w, h))
        if self.pose is not None:
            self._attach_keypoints(frame_bgr, tracks)
        return tracks

    def _attach_keypoints(self, frame, tracks):
        """Match pose detections to person tracks by IoU and attach keypoints."""
        res = self.pose(frame, verbose=False, device=self.device, conf=self.conf, imgsz=self.imgsz)
        if not res or res[0].keypoints is None or res[0].boxes is None:
            return
        pboxes = res[0].boxes.xyxy.cpu().numpy()
        kps = res[0].keypoints.data.cpu().numpy()  # (n, 17, 3)
        for t in tracks:
            if t["class"] != "person":
                continue
            best, best_iou = None, 0.3
            for i, pb in enumerate(pboxes):
                iou = _iou(t["bbox"], pb)
                if iou > best_iou:
                    best, best_iou = i, iou
            if best is not None:
                t["kpts"] = kps[best].tolist()


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ----------------------------------------------------------------------------- Motion (no-ML fallback)
class MotionDetector:
    """
    MOG2 background subtraction + greedy IoU/centroid tracker.
    Coarse classes from blob shape: tall -> person, wide & large -> car. Demo/testing only.
    """
    name = "motion"

    def __init__(self, min_area_frac: float = 0.0008, max_missed: int = 8):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=32, detectShadows=True)
        self.min_area_frac = min_area_frac
        self.max_missed = max_missed
        self._next_id = 1
        self._tracks: Dict[int, Dict] = {}   # id -> {bbox, missed, hits}
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def process_frame(self, frame_bgr: np.ndarray) -> List[Dict]:
        h, w = frame_bgr.shape[:2]
        small = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
        mask = self.bg.apply(small)
        mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]   # drop shadows (value 127)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, iterations=1)
        mask = cv2.dilate(mask, self._kernel, iterations=2)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw * bh < self.min_area_frac * w * h:
                continue
            dets.append((float(x), float(y), float(x + bw), float(y + bh)))
        return self._assign(dets, w, h)

    def _assign(self, dets, w, h) -> List[Dict]:
        unmatched = set(range(len(dets)))
        pairs = []
        for tid, tr in self._tracks.items():
            for di in unmatched:
                iou = _iou(tr["bbox"], dets[di])
                if iou > 0.1:
                    pairs.append((iou, tid, di))
        used = set()
        for iou, tid, di in sorted(pairs, reverse=True):
            if tid in used or di not in unmatched:
                continue
            used.add(tid)
            unmatched.discard(di)
            self._tracks[tid].update(bbox=dets[di], missed=0, hits=self._tracks[tid]["hits"] + 1)
        for tid in list(self._tracks):
            if tid not in used:
                self._tracks[tid]["missed"] += 1
                if self._tracks[tid]["missed"] > self.max_missed:
                    del self._tracks[tid]
        for di in unmatched:
            self._tracks[self._next_id] = {"bbox": dets[di], "missed": 0, "hits": 1}
            self._next_id += 1
        out = []
        for tid, tr in self._tracks.items():
            if tr["missed"] == 0 and tr["hits"] >= 3:   # confirmed, visible this frame
                b = tr["bbox"]
                bw, bh = b[2] - b[0], b[3] - b[1]
                is_vehicle = bw > 1.4 * bh and bw * bh > 0.02 * w * h
                out.append(make_track(tid, "car" if is_vehicle else "person", 0.5, b, w, h))
        return out


def build_detector(cfg: Dict[str, Any]):
    """Create a detector from config["model"]; falls back to MotionDetector if YOLO is unavailable."""
    m = cfg["model"]
    backend = m.get("backend", "auto")
    if backend in ("auto", "yolo"):
        try:
            return YoloDetector(m["name"], m["conf"], m["imgsz"], m["device"], m["tracker"], m.get("pose_model"))
        except Exception as e:  # ImportError, missing weights, offline download...
            if backend == "yolo":
                raise
            print(f"[IBVAP] YOLO unavailable ({type(e).__name__}: {e}) -> using MotionDetector demo backend")
    return MotionDetector()
