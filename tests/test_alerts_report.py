import os
import tempfile
import time
import unittest

import numpy as np

from ibvap.alerts import AlertStore
from ibvap.report import build_incident_report, build_summary_report


def alert(i="a1", sev="CRITICAL", track=1, wall=None, **kw):
    d = {"id": i, "type": "PERIMETER_BREACH", "severity": sev, "track_id": track, "class_name": "person",
         "zone": "Fence <A&B>", "timestamp": 12.5, "wall_time": wall or time.time(), "camera_id": "CAM_01",
         "sector": "S4", "message": "PERIMETER_BREACH: PERSON crossed Fence", "plate": "DL 01 AB 1234"}
    d.update(kw)
    return d


class Store(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = AlertStore({"db_path": os.path.join(self.tmp.name, "t.db"),
                                 "snapshot_dir": os.path.join(self.tmp.name, "snaps"), "dedupe_sec": 2.0})

    def tearDown(self):
        self.tmp.cleanup()

    def test_add_get_snapshot_extra(self):
        frame = np.full((90, 160, 3), 128, np.uint8)
        self.assertTrue(self.store.add(alert(), snapshot=frame))
        row = self.store.get("a1")
        self.assertEqual(row["severity"], "CRITICAL")
        self.assertEqual(row["extra"]["plate"], "DL 01 AB 1234")
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, "snaps", row["snapshot"])))

    def test_dedupe_same_track_within_window(self):
        t = time.time()
        self.assertTrue(self.store.add(alert("a1", wall=t)))
        self.assertFalse(self.store.add(alert("a2", wall=t + 1)))
        self.assertTrue(self.store.add(alert("a3", wall=t + 5)))
        self.assertTrue(self.store.add(alert("a4", track=2, wall=t + 5.1)))

    def test_filters_stats_ack(self):
        t = time.time()
        self.store.add(alert("a1", "CRITICAL", 1, t))
        self.store.add(alert("a2", "LOW", 2, t, type="PLATE_READ"))
        self.assertEqual([r["id"] for r in self.store.recent(min_severity="HIGH")], ["a1"])
        s = self.store.stats()
        self.assertEqual((s["total"], s["by_severity"]["LOW"], s["by_type"]["PLATE_READ"]), (2, 1, 1))
        self.assertTrue(self.store.acknowledge("a1", "checked"))
        self.assertEqual(self.store.get("a1")["acknowledged"], 1)
        self.assertFalse(self.store.acknowledge("nope"))

    def test_subscribe_receives_alerts(self):
        q = self.store.subscribe()
        self.store.add(alert())
        self.assertEqual(q.get_nowait()["id"], "a1")

    def test_pdf_reports(self):
        frame = np.full((360, 640, 3), 100, np.uint8)
        self.store.add(alert(), snapshot=frame)
        row = self.store.get("a1")
        p = build_incident_report(row, os.path.join(self.tmp.name, "r.pdf"), self.store.snapshot_dir)
        self.assertTrue(open(p, "rb").read(5) == b"%PDF-")
        p2 = build_summary_report(self.store.recent(), self.store.stats(), os.path.join(self.tmp.name, "s.pdf"))
        self.assertGreater(os.path.getsize(p2), 1000)


if __name__ == "__main__":
    unittest.main()
