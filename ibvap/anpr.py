"""
Automatic Number Plate Recognition (ANPR) for Indian checkpoint vehicles.

Pipeline: locate plate in vehicle crop -> preprocess variants -> OCR (EasyOCR / Tesseract)
-> Indian-format correction (O/0, I/1 ... by position) -> per-track majority voting
-> watchlist match.
"""

import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

VALID_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP", "HR",
    "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD",
    "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB", "BH",
}

# Standard civilian: SS DD [A-Z]{1,3} DDDD ; Bharat series: DD BH DDDD [A-Z]{1,2}
CIVILIAN_RE = re.compile(r"^([A-Z]{2})([0-9]{1,2})([A-Z]{1,3})([0-9]{4})$")
BHARAT_RE = re.compile(r"^([0-9]{2})(BH)([0-9]{4})([A-Z]{1,2})$")
MILITARY_RE = re.compile(r"^(?:\^|↑)?([0-9]{2})([A-Z])([0-9]{6})[A-Z]?$")

TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7"}
TO_ALPHA = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "4": "A", "7": "T"}

DEMO_WATCHLIST = {   # SYNTHETIC demo data - replace with the real list in data/plate_watchlist.json
    "DL01AB1234": "DEMO: suspect vehicle - smuggling watch",
    "MH12DE1433": "DEMO: unregistered dumper - unauthorized border transit",
    "UP14BC9999": "DEMO: flagged by special cell",
}


def clean_text(raw: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (raw or "").upper())


def _fix(s: str, kinds: str) -> Optional[Tuple[str, int]]:
    """Positional correction. kinds: 'A' (letter) / 'D' (digit) per char. Returns (text, n_corrections)."""
    out, cost = [], 0
    for ch, k in zip(s, kinds):
        if k == "D":
            if ch.isdigit():
                out.append(ch)
            elif ch in TO_DIGIT:
                out.append(TO_DIGIT[ch]); cost += 1
            else:
                return None
        else:
            if ch.isalpha():
                out.append(ch)
            elif ch in TO_ALPHA:
                out.append(TO_ALPHA[ch]); cost += 1
            else:
                return None
    return "".join(out), cost


def parse_plate(raw: str) -> Optional[Tuple[str, str]]:
    """
    Parse noisy OCR text into (formatted_plate, type) or None.
    Every window of the text is tried with positional character correction (O/0, I/1, Z/2 ...);
    the candidate needing the fewest corrections wins (ties: longest, then earliest).
    """
    s = clean_text(raw)
    if len(s) < 8:
        return None
    cands = []   # (cost, -length, start, formatted, type)
    for start in range(0, len(s) - 9):                      # Bharat series
        for ln in (10, 11):
            w = s[start:start + ln]
            if len(w) < ln:
                continue
            r = _fix(w, "DD" + "AA" + "DDDD" + "A" * (ln - 8))
            m = BHARAT_RE.match(r[0]) if r else None
            if m:
                cands.append((r[1], -ln, start, f"{m[1]} BH {m[3]} {m[4]}", "BHARAT_SERIES"))
    for start in range(0, len(s) - 7):                      # standard civilian
        for n_dist in (2, 1):
            for n_series in (1, 2, 3):
                ln = 2 + n_dist + n_series + 4
                w = s[start:start + ln]
                if len(w) < ln:
                    continue
                r = _fix(w, "AA" + "D" * n_dist + "A" * n_series + "DDDD")
                m = CIVILIAN_RE.match(r[0]) if r else None
                if m and m[1] in VALID_STATE_CODES:
                    cands.append((r[1], -ln, start, f"{m[1]} {m[2].zfill(2)} {m[3]} {m[4]}", "CIVILIAN"))
    if cands:
        best = min(cands)
        return best[3], best[4]
    m = MILITARY_RE.match(s)
    if m:
        return f"↑ {m[1]}{m[2]} {m[3]}", "MILITARY_DEFENCE"
    return None


# ----------------------------------------------------------------------------- OCR backends
class _NoOCR:
    name = "none"

    def read(self, img) -> List[Tuple[str, float]]:
        return []


class _TesseractOCR:
    name = "tesseract"

    def __init__(self):
        import pytesseract
        self.pt = pytesseract
        self.pt.get_tesseract_version()   # raises if binary missing
        self.cfg = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    def read(self, img) -> List[Tuple[str, float]]:
        data = self.pt.image_to_data(img, config=self.cfg, output_type=self.pt.Output.DICT)
        words = [(w, float(c)) for w, c in zip(data["text"], data["conf"]) if w.strip() and float(c) >= 0]
        if not words:
            return []
        text = "".join(w for w, _ in words)
        conf = sum(c for _, c in words) / len(words) / 100.0
        return [(text, conf)]


class _EasyOCR:
    name = "easyocr"

    def __init__(self):
        import easyocr
        self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def read(self, img) -> List[Tuple[str, float]]:
        res = self.reader.readtext(img, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", detail=1, paragraph=False)
        if not res:
            return []
        res = sorted(res, key=lambda r: r[0][0][0])        # left-to-right
        text = "".join(r[1] for r in res)
        conf = sum(r[2] for r in res) / len(res)
        return [(text, float(conf))]


def build_ocr(name: str = "auto"):
    order = {"auto": ["easyocr", "tesseract"], "easyocr": ["easyocr"], "tesseract": ["tesseract"], "none": []}[name]
    for n in order:
        try:
            return _EasyOCR() if n == "easyocr" else _TesseractOCR()
        except Exception:
            continue
    if name != "none":
        print("[IBVAP] No OCR backend available (pip install easyocr, or install tesseract). ANPR will only locate plates.")
    return _NoOCR()


# ----------------------------------------------------------------------------- engine
class ANPREngine:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None, ocr=None):
        cfg = cfg or {}
        self.min_votes = int(cfg.get("min_votes", 3))
        self.ocr = ocr if ocr is not None else build_ocr(cfg.get("ocr_backend", "auto"))
        self.watchlist_plates = self._load_watchlist(cfg.get("watchlist_file"))
        self.scanned_records: List[Dict] = []
        self._votes: Dict[int, Counter] = {}
        self._confirmed: Dict[int, str] = {}

    @staticmethod
    def _load_watchlist(path: Optional[str]) -> Dict[str, str]:
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return {clean_text(k): v for k, v in json.load(f).items()}
            except Exception as e:
                print(f"[IBVAP] Bad plate watchlist {path}: {e}")
        return dict(DEMO_WATCHLIST)

    # -- compat helpers
    clean_text = staticmethod(clean_text)

    def validate_and_format_plate(self, raw: str) -> Optional[Tuple[str, str]]:
        return parse_plate(raw)

    def plate_for_track(self, track_id: int) -> Optional[str]:
        return self._confirmed.get(track_id)

    # -- plate localisation
    def locate_plate_region(self, vehicle_crop: np.ndarray) -> Optional[np.ndarray]:
        """Find the most plate-like rectangle in the lower part of the vehicle crop."""
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None
        h, w = vehicle_crop.shape[:2]
        y0 = int(h * 0.4)
        lower = vehicle_crop[y0:, :]
        gray = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)   # dark characters on light plate
        grad = np.absolute(cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1))
        lo, hi = float(grad.min()), float(grad.max())
        if hi <= lo:
            return None
        grad = (255 * (grad - lo) / (hi - lo)).astype("uint8")
        grad = cv2.GaussianBlur(grad, (5, 5), 0)
        _, th = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
        th = cv2.erode(th, None, iterations=2)
        th = cv2.dilate(th, None, iterations=2)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_score = None, 0.0
        for c in cnts:
            x, y, cw, ch = cv2.boundingRect(c)
            aspect = cw / max(1.0, ch)
            if not (2.0 <= aspect <= 6.0) or cw < max(40, 0.15 * w) or ch < 12:
                continue
            score = cw * ch * (1.0 - min(abs(aspect - 4.0) / 4.0, 0.9))
            if score > best_score:
                best, best_score = (x, y, cw, ch), score
        if best is None:
            return None
        x, y, cw, ch = best
        m = 4
        return lower[max(0, y - m):min(lower.shape[0], y + ch + m), max(0, x - m):min(lower.shape[1], x + cw + m)]

    @staticmethod
    def preprocess_variants(plate: np.ndarray) -> List[np.ndarray]:
        """Several upscaled/binarised versions - OCR is run on each, best parse wins."""
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY) if plate.ndim == 3 else plate
        scale = max(1.0, 80.0 / max(1, gray.shape[0]))
        gray = cv2.resize(gray, None, fx=scale * 2, fy=scale * 2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.copyMakeBorder(gray, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
        _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if otsu.mean() < 127:                  # make text dark on light
            otsu = 255 - otsu
        return [gray, clahe, otsu]

    def read_plate_image(self, plate: np.ndarray) -> Optional[Dict]:
        """OCR a tight plate crop. Returns {plate, type, confidence, raw} or None."""
        best = None
        for variant in self.preprocess_variants(plate):
            for text, conf in self.ocr.read(variant):
                parsed = parse_plate(text)
                if parsed and (best is None or conf > best["confidence"]):
                    best = {"plate": parsed[0], "type": parsed[1], "confidence": round(conf, 3), "raw": text}
            if best and best["confidence"] >= 0.8:      # confident enough - skip the other variants
                break
        return best

    def process_vehicle(self, vehicle_crop: np.ndarray, track_id: int, now: float = 0.0) -> Optional[Dict]:
        """
        Try to read the plate of one tracked vehicle. A plate is *confirmed* after ``min_votes``
        agreeing reads across frames (suppresses OCR flicker).
        """
        if track_id in self._confirmed:
            return None
        plate_crop = self.locate_plate_region(vehicle_crop)
        if plate_crop is None or plate_crop.size == 0:
            return None
        read = self.read_plate_image(plate_crop)
        if not read:
            return None
        votes = self._votes.setdefault(track_id, Counter())
        votes[read["plate"]] += 1
        plate, n = votes.most_common(1)[0]
        if n < self.min_votes:
            return {"plate": plate, "type": read["type"], "votes": n, "confirmed_now": False,
                    "is_flagged": False, "reason": ""}
        self._confirmed[track_id] = plate
        rec = self.register_scan(plate, read["type"], track_id=track_id, now=now)
        rec.update(votes=n, confirmed_now=True)
        return rec

    def register_scan(self, plate: str, p_type: str = "STANDARD", vehicle_class: str = "car",
                      track_id: int = -1, now: float = 0.0) -> Dict:
        key = clean_text(plate)
        flagged = key in self.watchlist_plates
        rec = {"plate": plate, "type": p_type, "vehicle_class": vehicle_class, "track_id": track_id,
               "timestamp": now, "is_flagged": flagged,
               "reason": self.watchlist_plates.get(key, "Clear - authorized transit")}
        self.scanned_records.append(rec)
        return rec

    def register_manual_scan(self, raw_plate: str, vehicle_class: str = "car") -> Dict:
        """Register a verified / manually keyed plate."""
        parsed = parse_plate(raw_plate)
        formatted, p_type = parsed if parsed else (raw_plate.upper().strip(), "STANDARD")
        return self.register_scan(formatted, p_type, vehicle_class)

    def forget_track(self, track_id: int):
        self._votes.pop(track_id, None)
        self._confirmed.pop(track_id, None)
