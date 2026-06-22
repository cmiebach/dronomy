"""Matcher interface, shared result type, and homography estimation."""
from __future__ import annotations

import abc
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MatchResult:
    """Outcome of matching a drone frame (src) to a reference tile (dst)."""
    src_pts: np.ndarray            # Nx2 keypoints in the drone frame
    dst_pts: np.ndarray            # Nx2 corresponding keypoints in the reference
    homography: np.ndarray | None  # 3x3 H mapping src -> dst, or None if failed
    inlier_mask: np.ndarray | None # N bool RANSAC inliers
    n_matches: int                 # raw correspondences before RANSAC

    @property
    def n_inliers(self) -> int:
        return int(self.inlier_mask.sum()) if self.inlier_mask is not None else 0

    @property
    def ok(self) -> bool:
        return self.homography is not None


def estimate_homography(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    reproj_threshold: float = 5.0,
    confidence: float = 0.999,
    min_inliers: int = 12,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Robust homography src->dst. Returns (H, inlier_mask) or (None, None).

    Uses MAGSAC++ (cv2.USAC_MAGSAC) — it weights correspondences by quality
    instead of a hard threshold, giving more stable, reproducible inlier counts
    near the low cross-modal floor — and falls back to plain RANSAC on the rare
    OpenCV builds/inputs where USAC errors out."""
    if len(src_pts) < 4:
        return None, None
    src = src_pts.reshape(-1, 1, 2).astype(np.float32)
    dst = dst_pts.reshape(-1, 1, 2).astype(np.float32)
    try:
        H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC,
                                     reproj_threshold, confidence=confidence,
                                     maxIters=10000)
    except cv2.error:
        H, mask = None, None
    if H is None:  # USAC can fail on degenerate sets; RANSAC is the safety net
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, reproj_threshold,
                                     confidence=confidence)
    if H is None:
        return None, None
    mask = mask.ravel().astype(bool)
    if int(mask.sum()) < min_inliers:
        return None, mask
    return H, mask


class Matcher(abc.ABC):
    @abc.abstractmethod
    def match(self, drone_bgr: np.ndarray, ref_rgb: np.ndarray) -> MatchResult:
        """Match a drone frame to a reference tile and estimate a homography."""
        raise NotImplementedError


def get_matcher(method: str, cfg=None) -> "Matcher":
    """Factory: 'classical' | 'loftr' | 'matchanything'."""
    method = method.lower()
    if method in ("classical", "sift", "orb", "akaze"):
        from .classical import ClassicalMatcher
        return ClassicalMatcher(cfg)
    if method in ("loftr", "deep", "superglue"):
        from .deep import DeepMatcher
        return DeepMatcher(cfg, model=method)
    if method in ("matchanything", "ma"):
        from .matchanything import MatchAnythingMatcher
        return MatchAnythingMatcher(cfg)
    raise ValueError(f"Unknown matcher: {method!r}")
