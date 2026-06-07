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
    """RANSAC homography src->dst. Returns (H, inlier_mask) or (None, None)."""
    if len(src_pts) < 4:
        return None, None
    H, mask = cv2.findHomography(
        src_pts.reshape(-1, 1, 2).astype(np.float32),
        dst_pts.reshape(-1, 1, 2).astype(np.float32),
        cv2.RANSAC, reproj_threshold, confidence=confidence,
    )
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
    """Factory: 'classical' | 'loftr' | 'superglue'."""
    method = method.lower()
    if method in ("classical", "sift", "orb", "akaze"):
        from .classical import ClassicalMatcher
        return ClassicalMatcher(cfg)
    if method in ("loftr", "deep", "superglue"):
        from .deep import DeepMatcher
        return DeepMatcher(cfg, model=method)
    raise ValueError(f"Unknown matcher: {method!r}")
