"""
Unified IBVAP Video Analytics Engine (one instance per camera).

Pipeline per frame:
  enhance -> detect+track -> behaviour -> tripwire/zones -> ANPR -> face watchlist -> threat level -> HUD

One engine == one camera: tracker state, tripwire history and bag history are per-camera,
so feeds never contaminate each other (v1 shared them).
"""

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .anpr import ANPREngine
from .behavior import BehaviorAnalyzer
from .config import load_config
from .detector import build_detector
from .enhancer import LowLightEnhancer
from .face_watch import FaceWatchlist
from .tripwire import TripwireManager

SEVERITY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
FONT = cv2.FONT_HERSHEY_SIMPLEX


class IBVAPEngine:
    def __init__(self, config: Optional[Dict[str, Any]] = None, camera_id: str = "CAM_01",
                 sector_name: str = "Sector 4 (Fence Perimeter)", detector=None, alert_store=None):
        self.cfg = config or load_config(None)
        self.camera_id = camera_id
        self.sector_name = sector_name
        self.alert_store = alert_store

        self.detector = detector or build_detector(self.cfg)
        self.tripwire = TripwireManager.from_config(self.cfg)
        self.behavior = BehaviorAnalyzer(self.cfg["behavior"])
        e = self.cfg["enhancer"]
        self.enhancer = LowLightEnhancer(clip_limit=e["clahe_clip"], low_light_threshold=e["low_light_threshold"])
        self.auto_low_light = e["auto_low_light"]
        self.anpr = ANPREngine(self.cfg["anpr"]) if self.cfg["anpr"]["enabled"] else None
        self.face_watch = FaceWatchlist(self.cfg["face"]) if self.cfg["face"]["enabled"] else None

        # telemetry
        self.frame_index = 0
        self.fps = 0.0
        self._last_wall = time.time()
        self.threat_level = "SECURE"
        self._critical_until = -1e9
        self._elevated_until = -1e9
        self.threat_hold_sec = 5.0
        self.recent_alerts: List[Dict] = []
        self.active_intruders = 0
        self.last_tracks: List[Dict] = []
        self.last_stage_ms: Dict[str, float] = {}

    # ------------------------------------------------------------------ main
    def process(self, frame_bgr: np.ndarray, ts: Optional[float] = None,
                night_mode: bool = False, thermal_mode: bool = False) -> Tuple[np.ndarray, List[Dict]]:
        """Returns (annotated_frame, new_alerts). ``ts`` is the stream clock in seconds."""
        wall = time.time()
        now = wall if ts is None else ts
        self.frame_index += 1
        dt = wall - self._last_wall
        self._last_wall = wall
        if dt > 0:
            inst = 1.0 / dt
            self.fps = round(inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst, 1)

        h, w = frame_bgr.shape[:2]
        stage: Dict[str, float] = {}
        t0 = time.perf_counter()

        def tick(name):
            nonlocal t0
            t1 = time.perf_counter()
            stage[name] = round((t1 - t0) * 1000, 2)
            t0 = t1

        # 1. enhancement
        working = frame_bgr
        if thermal_mode:
            working = self.enhancer.apply_tactical_thermal(frame_bgr)
        elif night_mode or (self.auto_low_light and self.enhancer.is_low_light(frame_bgr)):
            working = self.enhancer.enhance(frame_bgr, force=True)
        tick("enhance")

        # 2. detection + tracking
        tracks = self.detector.process_frame(working)
        tick("detect")

        new_alerts: List[Dict] = []

        # 3. behaviour
        new_alerts += self.behavior.check_anomalies(tracks, now, (w, h))
        tick("behavior")

        # 4. tripwire + zones
        new_alerts += self.tripwire.update(tracks, now)
        tick("tripwire")

        # 5. ANPR (vehicles) and face watchlist (people), throttled
        if self.anpr is not None:
            new_alerts += self._run_anpr(working, tracks, now)
        tick("anpr")
        if self.face_watch is not None:
            new_alerts += self._run_faces(working, tracks, now)
        tick("face")

        # enrich + threat level
        for a in new_alerts:
            a.setdefault("id", uuid.uuid4().hex[:12])
            a["camera_id"] = self.camera_id
            a["sector"] = self.sector_name
            a["wall_time"] = wall
        self._update_threat(new_alerts, now)

        self.recent_alerts = (new_alerts[::-1] + self.recent_alerts)[:50]
        self.last_tracks = tracks
        self.active_intruders = sum(1 for t in tracks if t["class"] == "person")

        # 6. HUD
        display = self._draw_tactical_hud(working.copy(), tracks, self.threat_level == "CRITICAL")
        tick("hud")
        self.last_stage_ms = stage

        # 7. persistence
        if self.alert_store is not None:
            for a in new_alerts:
                if not a.get("suppressed"):
                    self.alert_store.add(a, snapshot=display)
        return display, new_alerts

    # ------------------------------------------------------------------ modules
    def _crop(self, frame, bbox, pad=0.0):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        px, py = (x2 - x1) * pad, (y2 - y1) * pad
        x1, y1 = max(0, int(x1 - px)), max(0, int(y1 - py))
        x2, y2 = min(w, int(x2 + px)), min(h, int(y2 + py))
        return frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None

    def _run_anpr(self, frame, tracks, now) -> List[Dict]:
        out = []
        every = max(1, self.cfg["anpr"]["every_n_frames"])
        min_px = self.cfg["anpr"]["min_vehicle_px"]
        for t in tracks:
            if t["class"] not in ("car", "truck", "bus", "motorcycle") or t["id"] < 0:
                continue
            known = self.anpr.plate_for_track(t["id"])
            if known:
                t["plate"] = known
            if (self.frame_index + t["id"]) % every:
                continue
            x1, y1, x2, y2 = t["bbox"]
            if (x2 - x1) < min_px:
                continue
            crop = self._crop(frame, t["bbox"])
            reading = self.anpr.process_vehicle(crop, t["id"], now) if crop is not None else None
            if reading and reading.get("confirmed_now"):
                t["plate"] = reading["plate"]
                flagged = reading["is_flagged"]
                out.append({
                    "id": uuid.uuid4().hex[:12],
                    "type": "WATCHLIST_VEHICLE" if flagged else "PLATE_READ",
                    "severity": "CRITICAL" if flagged else "LOW",
                    "track_id": t["id"], "class_name": t["class"], "timestamp": now, "bbox": t["bbox"],
                    "plate": reading["plate"], "plate_type": reading["type"], "reason": reading["reason"],
                    "message": (f"WATCHLIST VEHICLE {reading['plate']}: {reading['reason']}" if flagged
                                else f"Plate read: {reading['plate']} ({reading['type']})")})
        return out

    def _run_faces(self, frame, tracks, now) -> List[Dict]:
        out = []
        every = max(1, self.cfg["face"]["every_n_frames"])
        for t in tracks:
            if t["class"] != "person" or t["id"] < 0 or (self.frame_index + t["id"]) % every:
                continue
            match = self.face_watch.identify_person(frame, t["bbox"], t["id"], now)
            if match:
                t["face_match"] = match["name"]
                out.append({
                    "id": uuid.uuid4().hex[:12], "type": "WATCHLIST_FACE",
                    "severity": match.get("risk_level", "HIGH") if match.get("risk_level") in SEVERITY_RANK else "HIGH",
                    "track_id": t["id"], "class_name": "person", "timestamp": now, "bbox": t["bbox"],
                    "poi_id": match["poi_id"], "similarity": match["score"],
                    "message": f"WATCHLIST MATCH: {match['name']} ({match['category']}) score {match['score']:.2f}"})
        return out

    def _update_threat(self, alerts: List[Dict], now: float):
        for a in alerts:
            if a.get("suppressed"):
                continue
            rank = SEVERITY_RANK.get(a["severity"], 0)
            if rank >= 3:
                self._critical_until = now + self.threat_hold_sec
            elif rank == 2:
                self._elevated_until = now + self.threat_hold_sec
        if now < self._critical_until:
            self.threat_level = "CRITICAL"
        elif now < self._elevated_until:
            self.threat_level = "ELEVATED"
        else:
            self.threat_level = "SECURE"

    # ------------------------------------------------------------------ HUD
    def _draw_tactical_hud(self, frame: np.ndarray, tracks: List[Dict], breached: bool) -> np.ndarray:
        h, w = frame.shape[:2]
        wire_color = (0, 0, 255) if breached else (0, 255, 255)

        for zone in self.tripwire.exclusion_zones:
            poly = np.array([(int(x * w), int(y * h)) for x, y in zone["polygon"]], np.int32)
            ov = frame.copy()
            cv2.fillPoly(ov, [poly], (0, 0, 180))
            cv2.addWeighted(ov, 0.12, frame, 0.88, 0, frame)
            cv2.polylines(frame, [poly], True, (0, 0, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, zone["name"].upper(), (poly[0][0] + 4, poly[0][1] + 16), FONT, 0.42,
                        (80, 80, 255), 1, cv2.LINE_AA)

        for wire in self.tripwire.tripwires:
            p1 = (int(wire["p1"][0] * w), int(wire["p1"][1] * h))
            p2 = (int(wire["p2"][0] * w), int(wire["p2"][1] * h))
            cv2.line(frame, p1, p2, wire_color, 3, cv2.LINE_AA)
            cv2.putText(frame, f"VIRTUAL TRIPWIRE: {wire['name'].upper()}", (p1[0], max(12, p1[1] - 8)),
                        FONT, 0.5, wire_color, 2, cv2.LINE_AA)

        for t in tracks:
            x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
            cls, tid, conf = t["class"], t["id"], int(t["conf"] * 100)
            if cls == "person":
                pose = t.get("pose") or "STANDING"
                if pose in ("CRAWLING", "CROUCHING"):
                    color, label = (0, 0, 255), f"SUSPECT #{tid} [{pose}] {conf}%"
                else:
                    color, label = (0, 140, 255), f"PERSON #{tid} {conf}%"
                if t.get("face_match"):
                    color, label = (0, 0, 255), f"WATCHLIST: {t['face_match']}"
            elif cls in ("car", "truck", "bus", "motorcycle"):
                color, label = (255, 200, 0), f"{cls.upper()} #{tid}"
                if t.get("plate"):
                    label += f" {t['plate']}"
            elif cls in ("cow", "dog", "horse", "sheep", "cat", "bird"):
                color, label = (180, 180, 180), f"WILDLIFE: {cls.upper()} (FILTERED)"
            else:
                color, label = (0, 255, 128), f"{cls.upper()} #{tid}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            c = max(2, min(15, (x2 - x1) // 4, (y2 - y1) // 4))
            for (ax, ay, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
                cv2.line(frame, (ax, ay), (ax + dx * c, ay), color, 4)
                cv2.line(frame, (ax, ay), (ax, ay + dy * c), color, 4)
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.45, 1)
            ty = y1 if y1 - th - 6 > 34 else y2 + th + 6          # keep label clear of the top banner
            cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 6, ty), color, -1)
            cv2.putText(frame, label, (x1 + 3, ty - 4), FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # top banner
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, 34), (12, 16, 24), -1)
        cv2.addWeighted(ov, 0.88, frame, 0.12, 0, frame)
        colors = {"CRITICAL": ((0, 0, 255), "THREAT: CRITICAL (BREACH)"),
                  "ELEVATED": ((0, 165, 255), "THREAT: ELEVATED"),
                  "SECURE": ((0, 255, 100), "STATUS: SECURE")}
        tc, text = colors[self.threat_level]
        cv2.circle(frame, (16, 17), 5, tc, -1)
        cv2.putText(frame, f"[{self.camera_id}]", (26, 22), FONT, 0.44, (240, 240, 240), 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(text, FONT, 0.44, 2)
        cv2.putText(frame, text, (max(w // 2 - tw // 2, 130), 22), FONT, 0.44, tc, 2, cv2.LINE_AA)
        stats = f"FPS {self.fps} | {len(tracks)} TGT"
        (sw, _), _ = cv2.getTextSize(stats, FONT, 0.40, 1)
        cv2.putText(frame, stats, (w - sw - 14, 22), FONT, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

        if breached and int(time.time() * 3) % 2 == 0:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
            msg = "PERIMETER BREACH IN PROGRESS"
            (mw, _), _ = cv2.getTextSize(msg, FONT, 0.75, 3)
            cv2.putText(frame, msg, ((w - mw) // 2, h - 25), FONT, 0.75, (0, 0, 255), 3, cv2.LINE_AA)
        return frame
