"""
Alert persistence + notification.

AlertStore: thread-safe SQLite log with a JPEG evidence snapshot per alert, de-duplication and
optional webhook / Telegram push for HIGH+ severity alerts (sent from a background thread so the
video loop never blocks).
"""

import json
import os
import queue
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

import cv2

SEVERITY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY, created REAL, stream_ts REAL, camera_id TEXT, sector TEXT,
    type TEXT, subtype TEXT, severity TEXT, track_id INTEGER, class_name TEXT, zone TEXT,
    message TEXT, snapshot TEXT, extra TEXT, acknowledged INTEGER DEFAULT 0, ack_note TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_sev ON alerts(severity);
"""
_KNOWN = {"id", "timestamp", "wall_time", "camera_id", "sector", "type", "subtype", "severity",
          "track_id", "class_name", "zone", "message", "suppressed"}


class AlertStore:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        c = cfg or {}
        self.db_path = c.get("db_path", "data/ibvap.db")
        self.snapshot_dir = c.get("snapshot_dir", "data/snapshots")
        self.dedupe_sec = float(c.get("dedupe_sec", 2.0))
        self.webhook_url = c.get("webhook_url")
        self.tg_token, self.tg_chat = c.get("telegram_bot_token"), c.get("telegram_chat_id")
        self.notify_min = SEVERITY_RANK.get(c.get("notify_min_severity", "HIGH"), 2)
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._recent: Dict[tuple, float] = {}
        self._q: "queue.Queue[Dict]" = queue.Queue(maxsize=200)
        self._subscribers: List[queue.Queue] = []
        if self.webhook_url or (self.tg_token and self.tg_chat):
            threading.Thread(target=self._notify_loop, daemon=True).start()

    # ------------------------------------------------------------------ write
    def add(self, alert: Dict, snapshot=None) -> bool:
        """Store an alert. Returns False if it was a duplicate within ``dedupe_sec``."""
        key = (alert.get("camera_id"), alert.get("type"), alert.get("track_id"), alert.get("zone"))
        now = alert.get("wall_time", time.time())
        with self._lock:
            if now - self._recent.get(key, -1e9) < self.dedupe_sec:
                return False
            self._recent[key] = now
            if len(self._recent) > 5000:
                self._recent = {k: t for k, t in self._recent.items() if now - t < 60}
            snap_name = None
            if snapshot is not None:
                snap_name = f"{time.strftime('%Y%m%d_%H%M%S', time.localtime(now))}_{alert['id']}.jpg"
                cv2.imwrite(os.path.join(self.snapshot_dir, snap_name), snapshot, [cv2.IMWRITE_JPEG_QUALITY, 85])
            extra = {k: v for k, v in alert.items() if k not in _KNOWN}
            self._db.execute(
                "INSERT OR REPLACE INTO alerts (id, created, stream_ts, camera_id, sector, type, subtype, severity,"
                " track_id, class_name, zone, message, snapshot, extra) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (alert["id"], now, alert.get("timestamp"), alert.get("camera_id"), alert.get("sector"),
                 alert.get("type"), alert.get("subtype"), alert.get("severity"), alert.get("track_id"),
                 alert.get("class_name"), alert.get("zone"), alert.get("message"), snap_name,
                 json.dumps(extra, default=str)))
            self._db.commit()
        alert["snapshot"] = snap_name
        for sub in list(self._subscribers):
            try:
                sub.put_nowait(alert)
            except queue.Full:
                pass
        if SEVERITY_RANK.get(alert.get("severity"), 0) >= self.notify_min and \
                (self.webhook_url or (self.tg_token and self.tg_chat)):
            try:
                self._q.put_nowait(alert)
            except queue.Full:
                pass
        return True

    def acknowledge(self, alert_id: str, note: str = "") -> bool:
        with self._lock:
            cur = self._db.execute("UPDATE alerts SET acknowledged=1, ack_note=? WHERE id=?", (note, alert_id))
            self._db.commit()
            return cur.rowcount > 0

    # ------------------------------------------------------------------ read
    def _rows(self, sql, args=()) -> List[Dict]:
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            if "extra" in d:
                d["extra"] = json.loads(d["extra"] or "{}")
            out.append(d)
        return out

    def recent(self, limit: int = 50, camera_id: Optional[str] = None, min_severity: Optional[str] = None,
               since: Optional[float] = None) -> List[Dict]:
        sql, args = "SELECT * FROM alerts WHERE 1=1", []
        if camera_id:
            sql += " AND camera_id=?"; args.append(camera_id)
        if min_severity:
            sevs = [s for s, r in SEVERITY_RANK.items() if r >= SEVERITY_RANK[min_severity]]
            sql += f" AND severity IN ({','.join('?' * len(sevs))})"; args += sevs
        if since:
            sql += " AND created>=?"; args.append(since)
        sql += " ORDER BY created DESC LIMIT ?"; args.append(limit)
        return self._rows(sql, args)

    def get(self, alert_id: str) -> Optional[Dict]:
        r = self._rows("SELECT * FROM alerts WHERE id=?", (alert_id,))
        return r[0] if r else None

    def stats(self, since: Optional[float] = None) -> Dict[str, Any]:
        since = since or 0
        by_sev = {r["severity"]: r["n"] for r in self._rows(
            "SELECT severity, COUNT(*) n FROM alerts WHERE created>=? GROUP BY severity", (since,))}
        by_type = {r["type"]: r["n"] for r in self._rows(
            "SELECT type, COUNT(*) n FROM alerts WHERE created>=? GROUP BY type", (since,))}
        return {"total": sum(by_sev.values()), "by_severity": by_sev, "by_type": by_type}

    def subscribe(self) -> "queue.Queue":
        q: queue.Queue = queue.Queue(maxsize=100)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q):
        if q in self._subscribers:
            self._subscribers.remove(q)

    # ------------------------------------------------------------------ notify
    def _notify_loop(self):
        while True:
            a = self._q.get()
            text = f"[IBVAP {a.get('severity')}] {a.get('camera_id')} - {a.get('message')}"
            try:
                if self.webhook_url:
                    body = json.dumps({"text": text, "alert": a}, default=str).encode()
                    req = urllib.request.Request(self.webhook_url, body, {"Content-Type": "application/json"})
                    urllib.request.urlopen(req, timeout=8).read()
                if self.tg_token and self.tg_chat:
                    url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
                    data = urllib.parse.urlencode({"chat_id": self.tg_chat, "text": text}).encode()
                    urllib.request.urlopen(urllib.request.Request(url, data), timeout=8).read()
            except Exception as e:
                print(f"[IBVAP] notification failed: {e}")
