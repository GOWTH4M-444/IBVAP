"""
Night-Vision & Low-Light Enhancer for Border Surveillance Feeds.
Adaptive CLAHE luminance boost, gamma correction and a thermal (Ironbow/Inferno) view.
"""

import cv2
import numpy as np


class LowLightEnhancer:
    def __init__(self, clip_limit: float = 3.0, tile_grid_size: tuple = (8, 8),
                 low_light_threshold: float = 70.0, gamma: float = 1.3):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.threshold = low_light_threshold
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        inv = 1.0 / gamma
        self._gamma_lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)

    def mean_luminance(self, frame_bgr: np.ndarray) -> float:
        """Cheap luminance estimate on a heavily downsampled frame."""
        small = frame_bgr[::8, ::8]
        if small.ndim == 3:
            small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return float(small.mean())

    def is_low_light(self, frame_bgr: np.ndarray, threshold: float = None) -> bool:
        return self.mean_luminance(frame_bgr) < (self.threshold if threshold is None else threshold)

    def enhance(self, frame_bgr: np.ndarray, force: bool = False) -> np.ndarray:
        """LAB-space CLAHE on the L channel (keeps colour), then mild gamma."""
        if not force and not self.is_low_light(frame_bgr):
            return frame_bgr
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        return cv2.LUT(out, self._gamma_lut)

    def apply_tactical_thermal(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Pseudo-thermal rendering (simulation of a multi-spectral feed)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(self._clahe.apply(gray), cv2.COLORMAP_INFERNO)
