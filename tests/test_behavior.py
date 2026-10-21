import unittest

from ibvap.behavior import BehaviorAnalyzer

W, H = 1000, 600


def person(tid, bbox, **kw):
    return dict(id=tid, **{"class": "person", "bbox": bbox, "conf": 0.9}, **kw)


def run(b, tracks_by_time):
    out = []
    for t, tracks in tracks_by_time:
        out += b.check_anomalies(tracks, t, (W, H))
    return out


class Posture(unittest.TestCase):
    def test_standing_never_alerts(self):
        b = BehaviorAnalyzer()
        a = run(b, [(i * 0.1, [person(1, (400, 200, 440, 300))]) for i in range(30)])
        self.assertEqual(a, [])

    def test_crawling_needs_persistence(self):
        b = BehaviorAnalyzer({"pose_confirm_frames": 6})
        crawl = (300, 300, 420, 340)
        a = run(b, [(i * 0.1, [person(1, crawl)]) for i in range(4)])
        self.assertEqual(a, [])                                       # one-off glitch -> nothing
        a = run(b, [(0.4 + i * 0.1, [person(1, crawl)]) for i in range(10)])
        self.assertEqual([x["subtype"] for x in a], ["CRAWLING"])
        self.assertEqual(a[0]["severity"], "CRITICAL")

    def test_edge_truncated_box_ignored(self):
        b = BehaviorAnalyzer()
        a = run(b, [(i * 0.1, [person(1, (0, 300, 120, 340))]) for i in range(20)])
        self.assertEqual(a, [])

    def test_keypoints(self):
        b = BehaviorAnalyzer()
        lying = [[0, 0, 0]] * 5 + [[100, 100, .9], [110, 100, .9]] + [[0, 0, 0]] * 4 + [[200, 105, .9], [210, 105, .9]] + [[0, 0, 0]] * 4
        standing = [[0, 0, 0]] * 5 + [[100, 100, .9], [110, 100, .9]] + [[0, 0, 0]] * 4 + [[102, 200, .9], [108, 200, .9]] + [[0, 0, 0]] * 2 + [[102, 330, .9], [108, 330, .9]]
        crouch = [[0, 0, 0]] * 5 + [[100, 100, .9], [110, 100, .9]] + [[0, 0, 0]] * 4 + [[102, 180, .9], [108, 180, .9]] + [[0, 0, 0]] * 2 + [[102, 220, .9], [108, 220, .9]]
        self.assertEqual(b.posture_from_keypoints(lying), "CRAWLING")
        self.assertEqual(b.posture_from_keypoints(standing), "STANDING")
        self.assertEqual(b.posture_from_keypoints(crouch), "CROUCHING")


class Bags(unittest.TestCase):
    def bag(self, tid=9):
        return {"id": tid, "class": "backpack", "bbox": (500, 400, 530, 430), "conf": .8}

    def test_abandoned_bag_alert(self):
        b = BehaviorAnalyzer({"bag_stationary_sec": 15})
        a = run(b, [(t, [self.bag()]) for t in range(0, 30)])
        self.assertEqual(len(a), 1)
        self.assertEqual(a[0]["type"], "UNATTENDED_BAGGAGE")
        self.assertGreaterEqual(a[0]["timestamp"], 15)

    def test_bag_with_owner_nearby_is_fine(self):
        b = BehaviorAnalyzer()
        owner = person(1, (520, 330, 560, 430))
        a = run(b, [(t, [self.bag(), owner]) for t in range(0, 60)])
        self.assertEqual([x for x in a if x["type"] == "UNATTENDED_BAGGAGE"], [])


class Crowd(unittest.TestCase):
    def test_crowd_alert_after_hold(self):
        b = BehaviorAnalyzer({"crowd_min_people": 4, "crowd_hold_sec": 3})
        group = [person(i, (400 + i * 30, 300, 425 + i * 30, 360)) for i in range(1, 6)]
        a = run(b, [(t * 0.5, group) for t in range(0, 20)])
        self.assertEqual([x["type"] for x in a], ["CROWD_FORMATION"])
        self.assertEqual(a[0]["count"], 5)

    def test_scattered_people_no_crowd(self):
        b = BehaviorAnalyzer({"crowd_min_people": 4})
        spread = [person(i, (i * 190, 300, i * 190 + 25, 360)) for i in range(1, 5)]
        self.assertEqual(run(b, [(t, spread) for t in range(10)]), [])


class Running(unittest.TestCase):
    def test_sustained_sprint_alerts_walking_does_not(self):
        b = BehaviorAnalyzer()
        sprint = [(i * 0.1, [person(1, (100 + i * 40, 300, 130 + i * 40, 360))]) for i in range(20)]   # 6.7 body-h/s
        walk = [(i * 0.1, [person(2, (100 + i * 5, 300, 130 + i * 5, 360))]) for i in range(20)]       # 0.8 body-h/s
        self.assertEqual([a["track_id"] for a in run(b, sprint + walk)], [1])


if __name__ == "__main__":
    unittest.main()
