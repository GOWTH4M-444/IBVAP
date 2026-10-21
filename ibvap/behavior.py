"""
Behavioral Anomaly & Tactical Activity Analyzer.

Detects: crawling/crouching infiltrators, unattended baggage, crowd formation and
rapid (running) movement.

Fixes over v1:
  * Posture needs to persist for N observations and uses a smoothed aspect ratio (far fewer
    false alarms from merged / truncated boxes). Boxes touching the frame edge are ignored.
  * Optional keypoint posture (YOLO-pose) when tracks carry ``kpts``.
  * All distances are relative to the frame size, so it works at any resolution.
  * Abandoned-bag logic tracks "time with no person nearby" instead of a one-instant check.
  * Clock is caller-supplied (``now``).
"""

import math
import uuid
from typing import Any, Dict, List, Optional, Tuple

BBox = Tuple[float, float, float, float]


def _center(b: BBox):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _foot(b: BBox):
    return ((b[0] + b[2]) / 2.0, b[3])


class BehaviorAnalyzer:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        c = dict(cfg or {})
        d = {
            "pose_method": "auto", "crouch_aspect": 0.75, "crawl_aspect": 1.25,
            "pose_confirm_frames": 6, "min_person_height_frac": 0.04, "edge_margin_frac": 0.01,
            "pose_cooldown_sec": 12.0, "bag_type": ["backpack", "suitcase", "handbag"],
            "bag_stationary_sec": 15.0, "bag_move_frac": 0.02, "bag_person_near_frac": 0.12,
            "bag_cooldown_sec": 20.0, "crowd_min_people": 5, "crowd_radius_frac": 0.12,
            "crowd_hold_sec": 3.0, "crowd_cooldown_sec": 30.0,
            "run_speed_body_heights_per_sec": 2.5, "run_confirm_frames": 4, "run_cooldown_sec": 15.0,
        }
        d.update(c)
        self.c = d
        self.bag_history: Dict[int, Dict] = {}
        self.alert_cooldowns: Dict[str, float] = {}
        self._pose_state: Dict[int, Dict] = {}      # track -> {ar, pose, streak}
        self._motion_state: Dict[int, Dict] = {}    # track -> {pos, t}
        self._crowd_since: Optional[float] = None

    # ------------------------------------------------------------------ posture
    def posture_from_aspect(self, bbox: BBox) -> str:
        x1, y1, x2, y2 = bbox
        ar = max(1.0, x2 - x1) / max(1.0, y2 - y1)
        return self._label_from_aspect(ar)

    def _label_from_aspect(self, ar: float) -> str:
        if ar >= self.c["crawl_aspect"]:
            return "CRAWLING"
        if ar >= self.c["crouch_aspect"]:
            return "CROUCHING"
        return "STANDING"

    def posture_from_keypoints(self, kpts) -> Optional[str]:
        """COCO-17 keypoints [[x, y, conf], ...]. Returns None if not enough confident points."""
        try:
            def pt(i):
                x, y, c = kpts[i]
                return (x, y) if c >= 0.3 else None
            ls, rs, lh, rh = pt(5), pt(6), pt(11), pt(12)
            la, ra = pt(15), pt(16)
            sh = [p for p in (ls, rs) if p]
            hp = [p for p in (lh, rh) if p]
            if not sh or not hp:
                return None
            sx = sum(p[0] for p in sh) / len(sh); sy = sum(p[1] for p in sh) / len(sh)
            hx = sum(p[0] for p in hp) / len(hp); hy = sum(p[1] for p in hp) / len(hp)
            dx, dy = abs(hx - sx), abs(hy - sy)
            if dx > dy * 1.2:               # torso is roughly horizontal -> lying / crawling
                return "CRAWLING"
            an = [p for p in (la, ra) if p]
            if an:
                ay = sum(p[1] for p in an) / len(an)
                torso = max(dy, 1.0)
                leg = abs(ay - hy)
                if leg / torso < 0.9:       # standing is ~1.5-2.2x torso length
                    return "CROUCHING"
            return "STANDING"
        except Exception:
            return None

    def analyze_human_pose(self, bbox: BBox, kpts=None) -> str:
        """Instantaneous posture label (used by the HUD)."""
        if kpts is not None and self.c["pose_method"] != "aspect":
            p = self.posture_from_keypoints(kpts)
            if p:
                return p
        return self.posture_from_aspect(bbox)

    def _confirmed_pose(self, p: Dict, frame_size, now: float) -> str:
        """Temporal-smoothed posture for a person track."""
        tid = p["id"]
        x1, y1, x2, y2 = p["bbox"]
        w, h = frame_size
        margin = self.c["edge_margin_frac"]
        truncated = (x1 <= margin * w or y1 <= margin * h or x2 >= (1 - margin) * w or y2 >= (1 - margin) * h)
        too_small = (y2 - y1) < self.c["min_person_height_frac"] * h
        st = self._pose_state.setdefault(tid, {"ar": None, "pose": "STANDING", "streak": 0, "seen": now})
        st["seen"] = now
        if truncated or too_small:
            st["streak"] = 0
            return "STANDING"
        kp_pose = self.posture_from_keypoints(p["kpts"]) if p.get("kpts") is not None and self.c["pose_method"] != "aspect" else None
        if kp_pose:
            label = kp_pose
        else:
            ar = max(1.0, x2 - x1) / max(1.0, y2 - y1)
            st["ar"] = ar if st["ar"] is None else 0.7 * st["ar"] + 0.3 * ar   # EMA
            label = self._label_from_aspect(st["ar"])
        if label == st["pose"]:
            st["streak"] += 1
        else:
            st["pose"], st["streak"] = label, 1
        return label if st["streak"] >= self.c["pose_confirm_frames"] else "STANDING"

    # ------------------------------------------------------------------ main
    def check_anomalies(self, tracks: List[Dict], current_time: float, frame_size: Tuple[int, int]) -> List[Dict]:
        """``frame_size`` is (width, height) in pixels. Returns new alerts."""
        w, h = frame_size
        diag = math.hypot(w, h)
        alerts: List[Dict] = []
        persons = [t for t in tracks if t["class"] == "person" and t["id"] is not None and t["id"] >= 0]
        bags = [t for t in tracks if t["class"] in self.c["bag_type"] and t["id"] is not None and t["id"] >= 0]

        # 1. crawling / crouching
        for p in persons:
            pose = self._confirmed_pose(p, frame_size, current_time)
            p["pose"] = pose
            if pose in ("CRAWLING", "CROUCHING"):
                if self._cooldown_ok(f"pose_{p['id']}_{pose}", current_time, self.c["pose_cooldown_sec"]):
                    alerts.append(self._alert(
                        "SUSPICIOUS_POSTURE", "CRITICAL" if pose == "CRAWLING" else "HIGH", p, current_time,
                        subtype=pose, message=f"SUSPICIOUS POSTURE: person #{p['id']} {pose} near perimeter"))

        # 2. rapid movement (running / evasion): smoothed speed in body-heights per second,
        #    must stay above the limit for several consecutive observations
        for p in persons:
            fx, fy = _foot(p["bbox"])
            body_h = max(1.0, p["bbox"][3] - p["bbox"][1])
            st = self._motion_state.get(p["id"])
            if st is None:
                self._motion_state[p["id"]] = {"pos": (fx, fy), "t": current_time, "ema": 0.0, "streak": 0}
                continue
            dt = current_time - st["t"]
            if dt < 0.05:
                continue
            inst = math.hypot(fx - st["pos"][0], fy - st["pos"][1]) / body_h / dt
            st["ema"] = 0.6 * st["ema"] + 0.4 * inst
            st["pos"], st["t"] = (fx, fy), current_time
            st["streak"] = st["streak"] + 1 if st["ema"] >= self.c["run_speed_body_heights_per_sec"] else 0
            if st["streak"] >= self.c["run_confirm_frames"] and \
                    self._cooldown_ok(f"run_{p['id']}", current_time, self.c["run_cooldown_sec"]):
                alerts.append(self._alert(
                    "RAPID_MOVEMENT", "MEDIUM", p, current_time,
                    message=f"RAPID MOVEMENT: person #{p['id']} moving at {st['ema']:.1f} body-heights/s"))

        # 3. crowd formation
        alerts.extend(self._check_crowd(persons, diag, current_time))

        # 4. unattended baggage
        alerts.extend(self._check_bags(bags, persons, diag, current_time))

        self._cleanup(tracks, current_time)
        return alerts

    # ------------------------------------------------------------------ helpers
    def _check_bags(self, bags, persons, diag, now) -> List[Dict]:
        out = []
        near_px = self.c["bag_person_near_frac"] * diag
        move_px = self.c["bag_move_frac"] * diag
        for b in bags:
            bid = b["id"]
            bx, by = _center(b["bbox"])
            info = self.bag_history.get(bid)
            if info is None:
                info = {"first_seen": now, "last_pos": (bx, by), "stationary_since": now,
                        "unattended_since": None, "last_seen": now}
                self.bag_history[bid] = info
            info["last_seen"] = now
            if math.hypot(bx - info["last_pos"][0], by - info["last_pos"][1]) > move_px:
                info["stationary_since"] = now
                info["last_pos"] = (bx, by)
                info["unattended_since"] = None
            has_owner = any(math.hypot(bx - _center(p["bbox"])[0], by - _center(p["bbox"])[1]) < near_px
                            for p in persons)
            if has_owner:
                info["unattended_since"] = None
            elif info["unattended_since"] is None:
                info["unattended_since"] = now
            if info["unattended_since"] is not None:
                dwell = now - max(info["unattended_since"], info["stationary_since"])
                if dwell >= self.c["bag_stationary_sec"] and \
                        self._cooldown_ok(f"bag_{bid}", now, self.c["bag_cooldown_sec"]):
                    out.append(self._alert(
                        "UNATTENDED_BAGGAGE", "CRITICAL", b, now,
                        message=f"HAZARD: unattended {b['class'].upper()} (ID #{bid}) without handler for {int(dwell)}s"))
        return out

    def _check_crowd(self, persons, diag, now) -> List[Dict]:
        n = len(persons)
        if n < self.c["crowd_min_people"]:
            self._crowd_since = None
            return []
        pts = [_foot(p["bbox"]) for p in persons]
        radius = self.c["crowd_radius_frac"] * diag
        # connected components by proximity
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i in range(n):
            for j in range(i + 1, n):
                if math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) <= radius:
                    parent[find(i)] = find(j)
        groups: Dict[int, List[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        biggest = max(groups.values(), key=len)
        if len(biggest) < self.c["crowd_min_people"]:
            self._crowd_since = None
            return []
        if self._crowd_since is None:
            self._crowd_since = now
        if now - self._crowd_since >= self.c["crowd_hold_sec"] and \
                self._cooldown_ok("crowd", now, self.c["crowd_cooldown_sec"]):
            cx = sum(pts[i][0] for i in biggest) / len(biggest)
            cy = sum(pts[i][1] for i in biggest) / len(biggest)
            return [{"id": uuid.uuid4().hex[:12], "type": "CROWD_FORMATION", "severity": "HIGH",
                     "track_id": -1, "class_name": "person", "timestamp": now,
                     "bbox": None, "position": (cx, cy), "count": len(biggest),
                     "message": f"CROWD FORMATION: {len(biggest)} people gathered"}]
        return []

    def _cooldown_ok(self, key: str, now: float, cooldown: float) -> bool:
        if now - self.alert_cooldowns.get(key, -1e9) >= cooldown:
            self.alert_cooldowns[key] = now
            return True
        return False

    def _cleanup(self, tracks, now):
        active = {t["id"] for t in tracks}
        for tid in list(self._pose_state):
            if tid not in active and now - self._pose_state[tid]["seen"] > 3.0:
                self._pose_state.pop(tid, None)
        for tid in list(self._motion_state):
            if tid not in active and now - self._motion_state[tid]["t"] > 3.0:
                self._motion_state.pop(tid, None)
        for bid in list(self.bag_history):
            if bid not in active and now - self.bag_history[bid]["last_seen"] > 30.0:
                self.bag_history.pop(bid, None)
        for k in [k for k, t in self.alert_cooldowns.items() if now - t > 300.0]:
            self.alert_cooldowns.pop(k, None)

    @staticmethod
    def _alert(type_, severity, t, now, message, subtype=None) -> Dict:
        a = {"id": uuid.uuid4().hex[:12], "type": type_, "severity": severity, "track_id": t["id"],
             "class_name": t["class"], "timestamp": now, "bbox": t["bbox"], "message": message}
        if subtype:
            a["subtype"] = subtype
        return a
