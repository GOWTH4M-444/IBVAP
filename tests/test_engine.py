import os
import tempfile
import unittest

import numpy as np

from ibvap.alerts import AlertStore
from ibvap.config import load_config
from ibvap.detector import make_track
from ibvap.engine import IBVAPEngine
from ibvap.face_watch import FaceWatchlist

W, H = 640, 480


class Scripted:
    """Detector that replays a list of per-frame track lists."""
    name = "scripted"

    def __init__(self, script):
        self.script, self.i = script, 0

    def process_frame(self, frame):
        t = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return t


def person(tid, x, y_foot, hgt=120):
    return make_track(tid, "person", 0.9, (x - 20, y_foot - hgt, x + 20, y_foot), W, H)


def cfg(**over):
    c = load_config(None)
    c["anpr"]["enabled"] = False
    c["model"]["backend"] = "motion"
    c["tripwires"] = [dict(id="w", name="Fence", p1=[0.1, 0.5], p2=[0.9, 0.5], direction="both")]
    c["zones"] = []
    c.update(over)
    return c


class EngineTests(unittest.TestCase):
    def frame(self):
        return np.full((H, W, 3), 120, np.uint8)

    def test_breach_sets_critical_holds_then_decays(self):
        script = [[person(1, 320, 200)], [person(1, 320, 320)]] + [[]] * 5
        e = IBVAPEngine(cfg(), detector=Scripted(script))
        out, a0 = e.process(self.frame(), 0.0)
        self.assertEqual((a0, e.threat_level), ([], "SECURE"))
        out, a1 = e.process(self.frame(), 0.5)
        self.assertEqual([a["type"] for a in a1], ["PERIMETER_BREACH"])
        self.assertEqual(a1[0]["camera_id"], "CAM_01")
        self.assertEqual(out.shape, (H, W, 3))
        e.process(self.frame(), 3.0); self.assertEqual(e.threat_level, "CRITICAL")   # held (v1 flickered)
        e.process(self.frame(), 6.0); self.assertEqual(e.threat_level, "SECURE")

    def test_cameras_are_isolated(self):
        s1 = [[person(1, 320, 200)], [person(1, 320, 320)]]
        e1 = IBVAPEngine(cfg(), "CAM_01", detector=Scripted(s1))
        e2 = IBVAPEngine(cfg(), "CAM_02", detector=Scripted([[person(1, 320, 400)]] * 2))   # same track id, other camera
        for t in (0.0, 0.5):
            e1.process(self.frame(), t); _, a2 = e2.process(self.frame(), t)
        self.assertEqual(a2, [])
        self.assertEqual(e1.recent_alerts[0]["camera_id"], "CAM_01")

    def test_alerts_persisted_with_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            store = AlertStore({"db_path": os.path.join(d, "x.db"), "snapshot_dir": os.path.join(d, "s")})
            e = IBVAPEngine(cfg(), detector=Scripted([[person(1, 320, 200)], [person(1, 320, 320)]]), alert_store=store)
            e.process(self.frame(), 0.0); e.process(self.frame(), 0.5)
            rows = store.recent()
            self.assertEqual(len(rows), 1)
            self.assertTrue(os.path.exists(os.path.join(d, "s", rows[0]["snapshot"])))

    def test_face_match_raises_watchlist_alert(self):
        c = cfg()
        c["face"]["enabled"] = True
        poi = np.ones(8, np.float32)
        e = IBVAPEngine(c, detector=Scripted([[person(1, 320, 400, 200)]] * 12))
        e.face_watch = FaceWatchlist({"match_threshold": 0.5}, embedder=lambda img: poi)
        e.face_watch.suspects = {"POI_9": {"name": "Test Subject", "category": "demo", "risk_level": "CRITICAL"}}
        e.face_watch.embeddings = {"POI_9": [poi]}
        types = []
        for i in range(8):
            _, al = e.process(self.frame(), i * 0.1)
            types += [a["type"] for a in al]
        self.assertEqual(types, ["WATCHLIST_FACE"])           # once, thanks to per-track cooldown
        self.assertEqual(e.threat_level, "CRITICAL")

    def test_motion_backend_runs_on_noise_frames(self):
        e = IBVAPEngine(cfg())
        rng = np.random.default_rng(0)
        for i in range(5):
            e.process(rng.integers(0, 255, (H, W, 3), dtype=np.uint8), i * 0.1)
        self.assertIn("detect", e.last_stage_ms)

    def test_night_and_thermal_modes(self):
        e = IBVAPEngine(cfg(), detector=Scripted([[]]))
        dark = np.full((H, W, 3), 15, np.uint8)
        self.assertTrue(e.enhancer.is_low_light(dark))
        out, _ = e.process(dark, 0.0); self.assertEqual(out.shape, dark.shape)
        out, _ = e.process(dark, 0.1, thermal_mode=True); self.assertEqual(out.shape, dark.shape)


class FaceMatching(unittest.TestCase):
    def test_threshold_and_best_match(self):
        a, b = np.array([1, 0, 0, 0], np.float32), np.array([0, 1, 0, 0], np.float32)
        f = FaceWatchlist({"match_threshold": 0.6}, embedder=lambda im: a)
        f.embeddings = {"A": [a], "B": [b]}
        self.assertEqual(f.match_embedding(np.array([0.9, 0.1, 0, 0], np.float32))[0], "A")
        self.assertIsNone(f.match_embedding(np.array([0.5, 0.5, 0.5, 0.5], np.float32)))

    def test_disabled_without_models(self):
        f = FaceWatchlist({"detector_model": "nope.onnx", "recognizer_model": "nope2.onnx"})
        self.assertFalse(f.enabled)
        self.assertIsNone(f.match_suspect(np.zeros((50, 50, 3), np.uint8)))


if __name__ == "__main__":
    unittest.main()
