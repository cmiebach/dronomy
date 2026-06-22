"""Classical feature matcher: SIFT/ORB/AKAZE + Lowe ratio test + RANSAC.

This is the baseline approach the brief asks us to compare against deep matchers.
SIFT is scale/rotation invariant, which matters because the drone altitude (scale)
and heading (rotation) relative to the north-up satellite tile are unknown.
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import Matcher, MatchResult, estimate_homography


class ClassicalMatcher(Matcher):
    def __init__(self, cfg=None):
        c = getattr(getattr(cfg, "matching", None), "classical", None) if cfg else None
        r = getattr(getattr(cfg, "matching", None), "ransac", None) if cfg else None
        self.detector_name = getattr(c, "detector", "SIFT").upper()
        self.max_features = getattr(c, "max_features", 8000)
        self.ratio = getattr(c, "ratio_test", 0.75)
        self.reproj = getattr(r, "reproj_threshold_px", 5.0)
        self.confidence = getattr(r, "confidence", 0.999)
        self.min_inliers = getattr(r, "min_inliers", 12)
        self._detector = self._build_detector()
        self._norm = cv2.NORM_HAMMING if self.detector_name == "ORB" else cv2.NORM_L2

    def _build_detector(self):
        if self.detector_name == "SIFT":
            return cv2.SIFT_create(nfeatures=self.max_features)
        if self.detector_name == "ORB":
            return cv2.ORB_create(nfeatures=self.max_features)
        if self.detector_name == "AKAZE":
            return cv2.AKAZE_create()
        raise ValueError(f"Unsupported detector: {self.detector_name}")

    @staticmethod
    def _gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return img
        # Accept BGR (drone frames) or RGB (reference) — luminance is the same.
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def match(self, drone_bgr: np.ndarray, ref_rgb: np.ndarray) -> MatchResult:
        g1, g2 = self._gray(drone_bgr), self._gray(ref_rgb)
        kp1, des1 = self._detector.detectAndCompute(g1, None)
        kp2, des2 = self._detector.detectAndCompute(g2, None)
        empty = MatchResult(np.empty((0, 2)), np.empty((0, 2)), None, None, 0)
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return empty

        bf = cv2.BFMatcher(self._norm)
        knn = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < self.ratio * n.distance]
        if len(good) < 4:
            return MatchResult(np.empty((0, 2)), np.empty((0, 2)), None, None, len(good))

        src = np.float32([kp1[m.queryIdx].pt for m in good])
        dst = np.float32([kp2[m.trainIdx].pt for m in good])
        H, mask = estimate_homography(src, dst, self.reproj, self.confidence, self.min_inliers)
        return MatchResult(src, dst, H, mask, len(good))
