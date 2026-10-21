"""
Watchlist Facial Recognition for border checkpoints.

Backend: OpenCV YuNet (face detection) + SFace (128-d embeddings), both small ONNX models run via
OpenCV itself - no extra Python packages. Download them with ``python scripts/download_models.py``.

Enrolment layout (all data is YOUR responsibility - use only lawfully held photos):
    data/watchlist/meta.json                 {"POI_001": {"name": ..., "category": ..., "risk_level": "HIGH"}}
    data/watchlist/POI_001/photo1.jpg ...    one or more clear frontal photos per person

Privacy note: biometric matching of real people needs legal authority, access control and audit
logging. Every match is written to the alert database (who/when/camera/score).
"""

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float32), b.ravel().astype(np.float32)
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d > 0 else 0.0


class FaceWatchlist:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None,
                 embedder: Optional[Callable[[np.ndarray], Optional[np.ndarray]]] = None):
        """``embedder(bgr_image) -> embedding|None`` can be injected (tests / other backends)."""
        cfg = cfg or {}
        self.threshold = float(cfg.get("match_threshold", 0.363))
        self.watchlist_dir = cfg.get("watchlist_dir", "data/watchlist")
        self.suspects: Dict[str, Dict] = {}
        self.embeddings: Dict[str, List[np.ndarray]] = {}
        self._cooldown: Dict[int, float] = {}
        self.enabled = False
        self._embed = embedder
        if embedder is None:
            self._embed = self._build_sface(cfg)
        if self._embed is not None:
            self.enabled = True
            self.load_watchlist()

    # ------------------------------------------------------------------ backend
    def _build_sface(self, cfg) -> Optional[Callable]:
        det_path = cfg.get("detector_model", "models/face_detection_yunet_2023mar.onnx")
        rec_path = cfg.get("recognizer_model", "models/face_recognition_sface_2021dec.onnx")
        if not (os.path.exists(det_path) and os.path.exists(rec_path)):
            print("[IBVAP] Face models missing -> face matching disabled. Run: python scripts/download_models.py")
            return None
        try:
            self._detector = cv2.FaceDetectorYN.create(det_path, "", (320, 320), 0.7, 0.3, 5000)
            self._recognizer = cv2.FaceRecognizerSF.create(rec_path, "")
        except Exception as e:
            print(f"[IBVAP] Could not initialise face models: {e}")
            return None

        def embed(img: np.ndarray) -> Optional[np.ndarray]:
            if img is None or img.size == 0:
                return None
            h, w = img.shape[:2]
            self._detector.setInputSize((w, h))
            _, faces = self._detector.detect(img)
            if faces is None or len(faces) == 0:
                return None
            face = max(faces, key=lambda f: f[2] * f[3])          # largest face
            aligned = self._recognizer.alignCrop(img, face)
            return self._recognizer.feature(aligned).ravel()
        return embed

    # ------------------------------------------------------------------ enrolment
    def load_watchlist(self):
        meta_path = os.path.join(self.watchlist_dir, "meta.json")
        if not os.path.exists(meta_path):
            return
        with open(meta_path, "r", encoding="utf-8") as f:
            self.suspects = json.load(f)
        for poi_id in self.suspects:
            folder = os.path.join(self.watchlist_dir, poi_id)
            if not os.path.isdir(folder):
                continue
            embs = []
            for fn in sorted(os.listdir(folder)):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    e = self._embed(cv2.imread(os.path.join(folder, fn)))
                    if e is not None:
                        embs.append(e)
                    else:
                        print(f"[IBVAP] No face found in {folder}/{fn}")
            if embs:
                self.embeddings[poi_id] = embs
        print(f"[IBVAP] Face watchlist: {len(self.embeddings)} / {len(self.suspects)} POIs enrolled")

    def enroll(self, poi_id: str, image_bgr: np.ndarray, meta: Optional[Dict] = None) -> bool:
        e = self._embed(image_bgr) if self._embed else None
        if e is None:
            return False
        self.embeddings.setdefault(poi_id, []).append(e)
        if meta:
            self.suspects[poi_id] = meta
        return True

    # ------------------------------------------------------------------ matching
    def match_embedding(self, emb: np.ndarray) -> Optional[Tuple[str, float]]:
        best_id, best = None, self.threshold
        for poi_id, embs in self.embeddings.items():
            s = max(cosine(emb, e) for e in embs)
            if s >= best:
                best_id, best = poi_id, s
        return (best_id, best) if best_id else None

    def match_suspect(self, face_crop: np.ndarray) -> Optional[Dict]:
        """Match a BGR image containing a face against the watchlist."""
        if not self.enabled or face_crop is None or face_crop.size == 0:
            return None
        emb = self._embed(face_crop)
        if emb is None:
            return None
        m = self.match_embedding(emb)
        if not m:
            return None
        poi_id, score = m
        info = self.suspects.get(poi_id, {})
        return {"poi_id": poi_id, "score": round(score, 3), "name": info.get("name", poi_id),
                "category": info.get("category", ""), "risk_level": info.get("risk_level", "HIGH"),
                "issuing_unit": info.get("issuing_unit", "")}

    @staticmethod
    def extract_head_region(person_bbox, frame_bgr: np.ndarray, frac: float = 0.35) -> Optional[np.ndarray]:
        """Upper part of a person box, padded sideways, upscaled for the face detector."""
        x1, y1, x2, y2 = [int(v) for v in person_bbox]
        h, w = frame_bgr.shape[:2]
        pw, ph = x2 - x1, y2 - y1
        if ph < 40 or pw < 15:
            return None
        x1, x2 = max(0, x1 - pw // 4), min(w, x2 + pw // 4)
        y1, y2c = max(0, y1), min(h, y1 + int(ph * frac))
        crop = frame_bgr[y1:y2c, x1:x2]
        if crop.size == 0:
            return None
        if crop.shape[0] < 160:
            s = 160 / crop.shape[0]
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        return crop

    def identify_person(self, frame_bgr: np.ndarray, person_bbox, track_id: int, now: float,
                        cooldown_sec: float = 30.0) -> Optional[Dict]:
        """Engine hook: returns a match once per track per cooldown."""
        if not self.enabled or not self.embeddings:
            return None
        if now - self._cooldown.get(track_id, -1e9) < cooldown_sec:
            return None
        head = self.extract_head_region(person_bbox, frame_bgr)
        m = self.match_suspect(head) if head is not None else None
        if m:
            self._cooldown[track_id] = now
        return m
