import unittest

from ibvap.tripwire import TripwireManager, point_in_polygon, segment_crossing

WIRE = dict(id="w", name="Fence", p1=(0.1, 0.5), p2=(0.9, 0.5))


def mgr(direction="both", **kw):
    return TripwireManager([dict(WIRE, direction=direction)], kw.pop("zones", []), **kw)


class Geometry(unittest.TestCase):
    def test_crossing_sign(self):
        self.assertEqual(segment_crossing((0.5, 0.4), (0.5, 0.6), (0.1, 0.5), (0.9, 0.5)), 1)
        self.assertEqual(segment_crossing((0.5, 0.6), (0.5, 0.4), (0.1, 0.5), (0.9, 0.5)), -1)

    def test_going_around_the_end_is_not_a_crossing(self):
        self.assertEqual(segment_crossing((0.95, 0.4), (0.95, 0.6), (0.1, 0.5), (0.9, 0.5)), 0)

    def test_polygon(self):
        sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
        self.assertTrue(point_in_polygon((0.5, 0.5), sq))
        self.assertFalse(point_in_polygon((1.5, 0.5), sq))
        self.assertFalse(point_in_polygon((0.5, 0.5), sq[:2]))


class Crossings(unittest.TestCase):
    def test_breach_critical_with_direction(self):
        m = mgr()
        self.assertEqual(m.update_track(1, (0.5, 0.4), "person", 0.0), [])
        a = m.update_track(1, (0.5, 0.6), "person", 0.1)
        self.assertEqual(len(a), 1)
        self.assertEqual((a[0]["type"], a[0]["severity"], a[0]["subtype"]), ("PERIMETER_BREACH", "CRITICAL", "in"))

    def test_direction_filter(self):
        m = mgr("in")
        m.update_track(1, (0.5, 0.6), "person", 0.0)
        self.assertEqual(m.update_track(1, (0.5, 0.4), "person", 0.1), [])      # "out" ignored
        m.update_track(2, (0.5, 0.4), "person", 0.0)
        self.assertEqual(len(m.update_track(2, (0.5, 0.6), "person", 0.1)), 1)  # "in" fires

    def test_cooldown_then_rearm(self):
        m = mgr(cross_cooldown_sec=10)
        m.update_track(1, (0.5, 0.4), "person", 0.0)
        self.assertEqual(len(m.update_track(1, (0.5, 0.6), "person", 1.0)), 1)
        m.update_track(1, (0.5, 0.4), "person", 2.0)
        self.assertEqual(m.update_track(1, (0.5, 0.6), "person", 3.0), [])      # inside cooldown
        again = m.update_track(1, (0.5, 0.4), "person", 20.0)       # cooldown over -> fires again
        self.assertEqual([a["subtype"] for a in again], ["out"])

    def test_animal_is_low_and_suppressed(self):
        m = mgr()
        m.update_track(1, (0.5, 0.4), "cow", 0.0)
        a = m.update_track(1, (0.5, 0.6), "cow", 0.1)[0]
        self.assertEqual((a["type"], a["severity"], a["suppressed"]), ("WILDLIFE_CROSSING", "LOW", True))

    def test_bag_crossing_ignored(self):
        m = mgr()
        m.update_track(1, (0.5, 0.4), "backpack", 0.0)
        self.assertEqual(m.update_track(1, (0.5, 0.6), "backpack", 0.1), [])

    def test_untracked_ids_skipped(self):
        m = mgr()
        t = {"id": -1, "class": "person", "foot_norm": (0.5, 0.4)}
        self.assertEqual(m.update([t], 0.0), [])
        self.assertEqual(m._last_pt, {})


class Zones(unittest.TestCase):
    ZONE = dict(id="z", name="Buffer", polygon=[(0, 0), (1, 0), (1, 1), (0, 1)], max_dwell_sec=10)

    def test_loitering_fires_after_dwell(self):
        """Regression: v1 trimmed history to 5 s so a 10 s dwell could never trigger."""
        m = TripwireManager([], [self.ZONE])
        fired = []
        for i in range(0, 16):
            fired += m.update_track(7, (0.5, 0.5), "person", float(i))
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0]["type"], "SUSPICIOUS_LOITERING")
        self.assertGreaterEqual(fired[0]["timestamp"], 10.0)

    def test_leaving_resets_dwell(self):
        m = TripwireManager([], [self.ZONE])
        fired = []
        for i in range(0, 9):
            fired += m.update_track(7, (0.5, 0.5), "person", float(i))
        for i in range(9, 13):                       # leaves for 4 s (> grace)
            fired += m.update_track(7, (2.0, 2.0), "person", float(i))
        for i in range(13, 20):                      # comes back; only 7 s of fresh dwell
            fired += m.update_track(7, (0.5, 0.5), "person", float(i))
        self.assertEqual(fired, [])

    def test_entry_alert(self):
        m = TripwireManager([], [dict(self.ZONE, alert_on_entry=True)])
        m.update_track(1, (2, 2), "person", 0.0)
        a = m.update_track(1, (0.5, 0.5), "person", 0.1)
        self.assertEqual(a[0]["type"], "ZONE_INTRUSION")


class Cleanup(unittest.TestCase):
    def test_state_is_pruned(self):
        m = mgr()
        for tid in range(100):
            m.update_track(tid, (0.5, 0.4), "person", 0.0)
            m.update_track(tid, (0.5, 0.6), "person", 0.1)
        self.assertTrue(m._cooldowns)
        m.cleanup([], 100.0)
        self.assertEqual((m._last_pt, m._last_seen, m._cooldowns), ({}, {}, {}))


if __name__ == "__main__":
    unittest.main()
