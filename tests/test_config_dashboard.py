import os
import tempfile
import unittest

import numpy as np

from ibvap.alerts import AlertStore
from ibvap.config import DEFAULTS, load_config


class Config(unittest.TestCase):
    def test_partial_yaml_merges_with_defaults(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("model:\n  conf: 0.5\nzones: []\n")
        c = load_config(f.name); os.unlink(f.name)
        self.assertEqual(c["model"]["conf"], 0.5)
        self.assertEqual(c["model"]["imgsz"], DEFAULTS["model"]["imgsz"])
        self.assertEqual(c["zones"], [])
        self.assertTrue(c["tripwires"])

    def test_repo_config_loads(self):
        root = os.path.dirname(os.path.dirname(__file__))
        c = load_config(os.path.join(root, "config.yaml"))
        self.assertEqual(c["cameras"][0]["id"], "CAM_01")

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_config("/no/such.yaml")


class Dashboard(unittest.TestCase):
    def test_endpoints(self):
        from ibvap.dashboard import create_app

        class FakeWorker:
            camera_id = "CAM_01"
            night_mode = thermal_mode = False

            def summary(self):
                return {"id": "CAM_01", "sector": "S", "status": "running", "fps": 10, "threat_level": "SECURE",
                        "persons": 0, "backend": "fake", "night_mode": self.night_mode, "thermal_mode": self.thermal_mode}

        with tempfile.TemporaryDirectory() as d:
            cfg = load_config(None)
            cfg["alerts"].update(db_path=os.path.join(d, "a.db"), snapshot_dir=os.path.join(d, "s"), report_dir=os.path.join(d, "r"))
            store = AlertStore(cfg["alerts"])
            store.add({"id": "x1", "type": "PERIMETER_BREACH", "severity": "CRITICAL", "track_id": 1, "class_name": "person",
                       "camera_id": "CAM_01", "message": "m", "timestamp": 1.0}, snapshot=np.zeros((90, 160, 3), np.uint8))
            c = create_app(cfg, [FakeWorker()], store).test_client()
            self.assertIn(b"IBVAP COMMAND CENTER", c.get("/").data)
            self.assertEqual(c.get("/api/cameras").json[0]["id"], "CAM_01")
            self.assertEqual(c.get("/api/alerts?min_severity=HIGH").json[0]["id"], "x1")
            self.assertEqual(c.get("/api/stats").json["total"], 1)
            self.assertEqual(c.post("/api/cameras/CAM_01/mode", json={"night": True}).json["night_mode"], True)
            self.assertEqual(c.post("/api/cameras/NOPE/mode", json={}).status_code, 404)
            self.assertTrue(c.get("/api/alerts/x1/report.pdf").data.startswith(b"%PDF"))
            self.assertTrue(c.get("/api/report/summary.pdf").data.startswith(b"%PDF"))
            self.assertEqual(c.post("/api/alerts/x1/ack", json={"note": "ok"}).json["ok"], True)
            snap = store.get("x1")["snapshot"]
            self.assertEqual(c.get(f"/snapshots/{snap}").status_code, 200)
            self.assertEqual(c.get("/snapshots/../a.db").status_code, 404)    # no path traversal


if __name__ == "__main__":
    unittest.main()
