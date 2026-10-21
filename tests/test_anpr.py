import unittest

import cv2
import numpy as np

from ibvap.anpr import ANPREngine, build_ocr, parse_plate


class Parse(unittest.TestCase):
    def test_clean_plates(self):
        self.assertEqual(parse_plate("DL01AB1234"), ("DL 01 AB 1234", "CIVILIAN"))
        self.assertEqual(parse_plate("MH12DE1433"), ("MH 12 DE 1433", "CIVILIAN"))
        self.assertEqual(parse_plate("KA 05 M 4321"), ("KA 05 M 4321", "CIVILIAN"))
        self.assertEqual(parse_plate("22BH1234AA"), ("22 BH 1234 AA", "BHARAT_SERIES"))

    def test_ocr_confusions_are_corrected_by_position(self):
        self.assertEqual(parse_plate("DLO1AB1Z34")[0], "DL 01 AB 1234")     # O->0, Z->2
        self.assertEqual(parse_plate("MH12DE14O3")[0], "MH 12 DE 1403")
        self.assertEqual(parse_plate("UP14BC9999")[0], "UP 14 BC 9999")

    def test_noise_around_plate(self):
        self.assertEqual(parse_plate("IND DL01AB1234 X")[0], "DL 01 AB 1234")

    def test_rejects_garbage_and_bad_state(self):
        self.assertIsNone(parse_plate("HELLOWORLD"))
        self.assertIsNone(parse_plate("ZZ01AB1234"))
        self.assertIsNone(parse_plate("1234"))


class FakeOCR:
    name = "fake"

    def __init__(self, seq):
        self.seq = list(seq)

    def read(self, img):
        return [(self.seq.pop(0), 0.9)] if self.seq else []


def plate_image(text="DL01AB1234"):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 64)
    im = Image.new("RGB", (560, 120), (245, 245, 245))
    ImageDraw.Draw(im).text((24, 20), text, fill=(10, 10, 10), font=font)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


class Engine(unittest.TestCase):
    def test_voting_and_watchlist(self):
        e = ANPREngine({"min_votes": 3}, ocr=FakeOCR(["DL01AB1234"] * 4))
        img = plate_image()
        self.assertEqual(e.read_plate_image(img)["plate"], "DL 01 AB 1234")
        # voting through process_vehicle needs a locatable plate; use a car-like crop
        vehicle = np.full((300, 400, 3), 90, np.uint8)
        vehicle[200:240, 100:300] = cv2.resize(img, (200, 40))
        for expected_confirm in (False, False, True):
            e.locate_plate_region = lambda crop: img
            r = e.process_vehicle(vehicle, 5, 1.0)
            self.assertEqual(r["confirmed_now"], expected_confirm)
        self.assertTrue(r["is_flagged"])
        self.assertEqual(e.plate_for_track(5), "DL 01 AB 1234")
        self.assertIsNone(e.process_vehicle(vehicle, 5, 2.0))   # already confirmed

    def test_voting_rejects_flicker(self):
        e = ANPREngine({"min_votes": 3}, ocr=FakeOCR(["DL01AB1234", "MH12DE1433", "DL01AB1234", "UP14BC9999", "DL01AB1234"]))
        e.locate_plate_region = lambda crop: plate_image()
        res = [e.process_vehicle(np.zeros((10, 10, 3), np.uint8), 1) for _ in range(5)]
        self.assertEqual([r["confirmed_now"] for r in res], [False, False, False, False, True])
        self.assertEqual(e.plate_for_track(1), "DL 01 AB 1234")

    def test_manual_scan(self):
        e = ANPREngine({}, ocr=FakeOCR([]))
        self.assertTrue(e.register_manual_scan("mh12de1433")["is_flagged"])
        self.assertFalse(e.register_manual_scan("KA05M4321")["is_flagged"])


@unittest.skipIf(build_ocr("tesseract").name == "none", "tesseract not installed")
class RealOCR(unittest.TestCase):
    def test_reads_rendered_plate_with_tesseract(self):
        e = ANPREngine({"ocr_backend": "tesseract"})
        r = e.read_plate_image(plate_image("MH12DE1433"))
        self.assertIsNotNone(r)
        self.assertEqual(r["plate"], "MH 12 DE 1433")


if __name__ == "__main__":
    unittest.main()
