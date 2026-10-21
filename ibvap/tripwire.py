"""
Virtual Tripwire & Geofencing Engine for Perimeter Intrusion Detection.

Fixes over v1:
  * Loitering actually fires (dwell is tracked explicitly, not derived from a 5 s history window).
  * Direction-aware crossings ("in" / "out" / "both").
  * Crossing is tested against the wire *segment*, using the object's foot point.
  * Only relevant classes raise breaches (bags etc. are ignored); animals are suppressed.
  * All state is keyed on a caller-supplied clock ``now`` (works for video files and live feeds).
  * Per-track state and the cooldown cache are pruned (no memory leak).
"""

import uuid
from typing import Any, Dict, List, Optional, Tuple

Point = Tuple[float, float]

ANIMAL_CLASSES = {"dog", "cat", "cow", "horse", "sheep", "bird"}


# --------------------------------------------------------------------------- geometry
def side_of_line(p1: Point, p2: Point, pt: Point) -> float:
    """Signed side of ``pt`` relative to the directed line p1->p2 (0 = on the line)."""
    return (p2[0] - p1[0]) * (pt[1] - p1[1]) - (p2[1] - p1[1]) * (pt[0] - p1[0])


def _sign(v: float, eps: float = 1e-12) -> int:
    return 1 if v > eps else (-1 if v < -eps else 0)


def segment_crossing(prev_pt: Point, cur_pt: Point, p1: Point, p2: Point) -> int:
    """
    Did the motion segment prev->cur cross the wire segment p1-p2?

    Returns +1 if it moved from side<0 to side>0, -1 for side>0 to side<0, else 0.
    A move that merely *touches* the line (ends exactly on it) is not counted; the next
    frame will resolve it, which avoids double triggers.
    """
    s_prev = _sign(side_of_line(p1, p2, prev_pt))
    s_cur = _sign(side_of_line(p1, p2, cur_pt))
    if s_prev == 0 or s_cur == 0 or s_prev == s_cur:
        return 0
    # The wire endpoints must lie on opposite sides of the motion segment
    # (otherwise the object went around the end of the wire).
    e1 = _sign(side_of_line(prev_pt, cur_pt, p1))
    e2 = _sign(side_of_line(prev_pt, cur_pt, p2))
    if e1 == e2:
        return 0
    return 1 if s_prev < 0 else -1


def segments_intersect(A: Point, B: Point, C: Point, D: Point) -> bool:
    """Backward-compatible helper: do segments AB and CD cross?"""
    return segment_crossing(A, B, C, D) != 0


def point_in_polygon(point: Point, polygon: List[Point]) -> bool:
    """Ray casting point-in-polygon."""
    x, y = point
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_int = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_int:
                inside = not inside
        j = i
    return inside


# --------------------------------------------------------------------------- manager
class TripwireManager:
    def __init__(self, tripwires: Optional[List[Dict]] = None, zones: Optional[List[Dict]] = None,
                 cross_cooldown_sec: float = 10.0, loiter_cooldown_sec: float = 15.0,
                 track_stale_sec: float = 2.0, zone_exit_grace_sec: float = 1.5):
        self.tripwires: List[Dict] = []
        self.exclusion_zones: List[Dict] = []
        for w in (tripwires or []):
            self.add_tripwire(**w)
        for z in (zones or []):
            self.add_zone(**z)

        self.cross_cooldown = cross_cooldown_sec
        self.loiter_cooldown = loiter_cooldown_sec
        self.track_stale_sec = track_stale_sec
        self.zone_exit_grace = zone_exit_grace_sec

        self._last_pt: Dict[int, Point] = {}
        self._last_seen: Dict[int, float] = {}
        # (track_id, zone_id) -> {"entered": t, "last_in": t}
        self._zone_state: Dict[Tuple[int, str], Dict[str, float]] = {}
        # (track_id, key) -> time of last alert
        self._cooldowns: Dict[Tuple[int, str], float] = {}

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "TripwireManager":
        t = cfg.get("tripwire", {})
        return cls(cfg.get("tripwires"), cfg.get("zones"),
                   cross_cooldown_sec=t.get("cross_cooldown_sec", 10.0),
                   loiter_cooldown_sec=t.get("loiter_cooldown_sec", 15.0),
                   track_stale_sec=t.get("track_stale_sec", 2.0),
                   zone_exit_grace_sec=t.get("zone_exit_grace_sec", 1.5))

    # -- configuration -------------------------------------------------------
    def add_tripwire(self, id: str, name: str, p1, p2, direction: str = "both",
                     classes: Optional[List[str]] = None, **_ignored):
        if direction not in ("both", "in", "out"):
            raise ValueError(f"direction must be both|in|out, got {direction!r}")
        self.tripwires.append({
            "id": id, "name": name, "p1": tuple(p1), "p2": tuple(p2), "direction": direction,
            "classes": set(classes or ["person", "bicycle", "motorcycle", "car", "bus", "truck"]),
        })

    def add_zone(self, id: str, name: str, polygon, max_dwell_sec: float = 10.0,
                 alert_on_entry: bool = False, **_ignored):
        self.exclusion_zones.append({
            "id": id, "name": name, "polygon": [tuple(p) for p in polygon],
            "max_dwell_sec": float(max_dwell_sec), "alert_on_entry": bool(alert_on_entry),
        })

    def set_tripwire(self, p1: Point, p2: Point, name: str = "Custom Perimeter Line",
                     direction: str = "both"):
        """Replace all wires with one custom line (normalized 0-1)."""
        self.tripwires = []
        self.add_tripwire("custom_tripwire_1", name, p1, p2, direction)
        self._cooldowns.clear()

    # -- main API -------------------------------------------------------------
    def update(self, tracks: List[Dict], now: float) -> List[Dict]:
        """Process all tracks of one frame. Uses ``foot_norm`` if present else ``center_norm``."""
        alerts: List[Dict] = []
        seen = []
        for t in tracks:
            tid = t.get("id", -1)
            if tid is None or tid < 0:
                continue  # untracked detections cannot be reasoned about over time
            pt = t.get("foot_norm") or t.get("center_norm")
            seen.append(tid)
            alerts.extend(self.update_track(tid, pt, t["class"], now))
        self.cleanup(seen, now)
        return alerts

    def update_track(self, track_id: int, point_norm: Point, class_name: str, now: float) -> List[Dict]:
        alerts: List[Dict] = []
        cls = class_name.lower()
        is_animal = cls in ANIMAL_CLASSES
        cur = (float(point_norm[0]), float(point_norm[1]))
        prev = self._last_pt.get(track_id)
        self._last_pt[track_id] = cur
        self._last_seen[track_id] = now

        # ---- tripwire crossings
        if prev is not None:
            for wire in self.tripwires:
                relevant = cls in wire["classes"] or is_animal
                if not relevant:
                    continue
                crossing = segment_crossing(prev, cur, wire["p1"], wire["p2"])
                if crossing == 0:
                    continue
                direction_name = "in" if crossing > 0 else "out"
                if wire["direction"] != "both" and wire["direction"] != direction_name:
                    continue
                key = (track_id, wire["id"])
                if now - self._cooldowns.get(key, -1e9) < self.cross_cooldown:
                    continue
                self._cooldowns[key] = now
                alert_type = "WILDLIFE_CROSSING" if is_animal else "PERIMETER_BREACH"
                alerts.append(self._alert(
                    alert_type, "LOW" if is_animal else "CRITICAL", track_id, cls, wire["name"], now, cur,
                    suppressed=is_animal, subtype=direction_name,
                    message=f"{alert_type}: {cls.upper()} (ID #{track_id}) crossed {wire['name']} ({direction_name.upper()})"))

        # ---- exclusion zones: entry + dwell
        for zone in self.exclusion_zones:
            zkey = (track_id, zone["id"])
            inside = point_in_polygon(cur, zone["polygon"])
            state = self._zone_state.get(zkey)
            if inside:
                if state is None:
                    state = {"entered": now, "last_in": now}
                    self._zone_state[zkey] = state
                    if zone["alert_on_entry"] and not is_animal:
                        alerts.append(self._alert(
                            "ZONE_INTRUSION", "HIGH", track_id, cls, zone["name"], now, cur,
                            message=f"ZONE INTRUSION: {cls.upper()} (ID #{track_id}) entered {zone['name']}"))
                state["last_in"] = now
                dwell = now - state["entered"]
                if dwell >= zone["max_dwell_sec"]:
                    key = (track_id, "loiter_" + zone["id"])
                    if now - self._cooldowns.get(key, -1e9) >= self.loiter_cooldown:
                        self._cooldowns[key] = now
                        alerts.append(self._alert(
                            "SUSPICIOUS_LOITERING", "MEDIUM" if is_animal else "HIGH", track_id, cls,
                            zone["name"], now, cur, suppressed=is_animal,
                            message=f"LOITERING: {cls.upper()} (ID #{track_id}) in {zone['name']} for {int(dwell)}s "
                                    f"(limit {zone['max_dwell_sec']:.0f}s)"))
            elif state is not None and now - state["last_in"] > self.zone_exit_grace:
                self._zone_state.pop(zkey, None)
        return alerts

    def cleanup(self, active_track_ids, now: float):
        """Drop state for tracks not seen for ``track_stale_sec``; prune cooldown cache."""
        active = set(active_track_ids)
        for tid in list(self._last_seen):
            if tid not in active and now - self._last_seen[tid] > self.track_stale_sec:
                self._last_seen.pop(tid, None)
                self._last_pt.pop(tid, None)
        for zkey in list(self._zone_state):
            tid = zkey[0]
            if tid not in self._last_seen:
                self._zone_state.pop(zkey, None)
        horizon = max(self.cross_cooldown, self.loiter_cooldown) * 2
        for k in [k for k, t in self._cooldowns.items() if now - t > horizon]:
            self._cooldowns.pop(k, None)

    @staticmethod
    def _alert(type_, severity, track_id, class_name, zone, now, pos, suppressed=False,
               message="", subtype=None) -> Dict:
        a = {"id": uuid.uuid4().hex[:12], "type": type_, "severity": severity, "track_id": track_id,
             "class_name": class_name, "zone": zone, "timestamp": now, "position": pos,
             "suppressed": suppressed, "message": message}
        if subtype:
            a["subtype"] = subtype
        return a
