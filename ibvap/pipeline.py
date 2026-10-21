"""
Camera ingestion: video files, webcams and RTSP/HTTP streams, with auto-reconnect.
CameraWorker runs one engine per camera in its own thread (multi-camera support).
"""

import os
import threading
import time
from typing import Any, Dict, Iterator, Optional, Tuple

import cv2
import numpy as np

from .engine import IBVAPEngine


def _open(source: str):
    src: Any = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if isinstance(src, str) and src.lower().startswith("rtsp"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class FrameSource:
    """Yields (frame, timestamp). Files use video time; live feeds use wall time."""

    def __init__(self, source: str, loop: bool = False, realtime: bool = False, max_width: int = 0,
                 reconnect_sec: float = 3.0):
        self.source, self.loop, self.realtime = source, loop, realtime
        self.max_width, self.reconnect_sec = max_width, reconnect_sec
        self.is_file = os.path.isfile(str(source))
        self.stop_event = threading.Event()

    def frames(self) -> Iterator[Tuple[np.ndarray, float]]:
        base_ts = 0.0
        while not self.stop_event.is_set():
            cap = _open(self.source)
            if not cap.isOpened():
                if self.is_file:
                    raise FileNotFoundError(f"Cannot open video: {self.source}")
                print(f"[IBVAP] cannot open {self.source}; retrying in {self.reconnect_sec}s")
                self.stop_event.wait(self.reconnect_sec)
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            if fps <= 1 or fps > 120:
                fps = 25.0
            idx, t_start = 0, time.time()
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                if self.max_width and frame.shape[1] > self.max_width:
                    s = self.max_width / frame.shape[1]
                    frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                if self.is_file:
                    ts = base_ts + idx / fps
                    if self.realtime:                       # pace to the video's own frame rate
                        wait = (idx / fps) - (time.time() - t_start)
                        if wait > 0:
                            time.sleep(wait)
                else:
                    ts = time.time()
                idx += 1
                yield frame, ts
            cap.release()
            if self.is_file:
                if self.loop:
                    base_ts += idx / fps + 1.0              # keep the clock monotonic across loops
                    continue
                return
            self.stop_event.wait(self.reconnect_sec)       # live feed dropped -> reconnect

    def stop(self):
        self.stop_event.set()


class CameraWorker(threading.Thread):
    """One camera = one thread = one engine. Latest annotated JPEG is available via ``latest_jpeg``."""

    def __init__(self, cam_cfg: Dict[str, Any], cfg: Dict[str, Any], alert_store=None,
                 realtime: bool = True, loop: bool = True):
        super().__init__(daemon=True, name=f"cam-{cam_cfg['id']}")
        self.cam = cam_cfg
        self.cfg = cfg
        self.camera_id = cam_cfg["id"]
        self.source = FrameSource(cam_cfg["source"], loop=cam_cfg.get("loop", loop), realtime=realtime,
                                  max_width=cam_cfg.get("max_width", 960))
        # per-camera overrides of tripwires / zones
        cam_full = dict(cfg)
        for k in ("tripwires", "zones"):
            if k in cam_cfg:
                cam_full[k] = cam_cfg[k]
        self.engine = IBVAPEngine(cam_full, self.camera_id, cam_cfg.get("sector", ""), alert_store=alert_store)
        self.night_mode = False
        self.thermal_mode = False
        self.status = "starting"
        self.error: Optional[str] = None
        self._jpeg: Optional[bytes] = None
        self._lock = threading.Lock()
        self._q = int(cfg["server"].get("jpeg_quality", 70))
        self._cond = threading.Condition()
        self._seq = 0

    def run(self):
        self.status = "running"
        try:
            for frame, ts in self.source.frames():
                out, _ = self.engine.process(frame, ts, self.night_mode, self.thermal_mode)
                ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, self._q])
                if ok:
                    with self._cond:
                        self._jpeg = buf.tobytes()
                        self._seq += 1
                        self._cond.notify_all()
            self.status = "ended"
        except Exception as e:                           # keep the dashboard alive if one camera dies
            self.status, self.error = "error", f"{type(e).__name__}: {e}"
            print(f"[IBVAP] {self.camera_id} crashed: {self.error}")

    def latest_jpeg(self) -> Optional[bytes]:
        return self._jpeg

    def wait_frame(self, last_seq: int, timeout: float = 2.0) -> Tuple[Optional[bytes], int]:
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            return self._jpeg, self._seq

    def stop(self):
        self.source.stop()

    def summary(self) -> Dict[str, Any]:
        e = self.engine
        return {"id": self.camera_id, "sector": self.cam.get("sector", ""), "status": self.status,
                "error": self.error, "fps": e.fps, "threat_level": e.threat_level, "targets": len(e.last_tracks),
                "persons": e.active_intruders, "night_mode": self.night_mode, "thermal_mode": self.thermal_mode,
                "backend": getattr(e.detector, "name", "?")}
