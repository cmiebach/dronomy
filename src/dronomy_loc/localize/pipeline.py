"""Turn a drone↔reference homography into a geographic pose estimate.

Given homography H (drone-frame pixels -> reference-tile pixels) and the reference
`GeoImage` (reference pixels -> lat/lon), we:
  1. project the drone-frame CENTER through H to get its reference pixel,
  2. convert that to (lat, lon)  ->  the drone's nadir ground position,
  3. derive yaw from how H rotates the frame's up-vector on the (north-up) tile,
  4. derive a scale (ground meters per drone pixel) as a rough altitude proxy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..matching.base import Matcher, MatchResult
from ..reference.geo import GeoImage


@dataclass
class PoseEstimate:
    ok: bool
    lat: float | None = None
    lon: float | None = None
    yaw_deg: float | None = None        # drone heading vs north (image-up vs tile-up)
    ground_m_per_px: float | None = None  # scale at the frame center (altitude proxy)
    n_inliers: int = 0
    n_matches: int = 0
    frame_index: int | None = None
    t_seconds: float | None = None


def _apply_H(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
    v = H @ np.array([x, y, 1.0])
    return v[0] / v[2], v[1] / v[2]


def pose_from_homography(H: np.ndarray, frame_shape, ref: GeoImage) -> PoseEstimate:
    h, w = frame_shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # 1-2. Frame center -> reference pixel -> lat/lon.
    rpx, rpy = _apply_H(H, cx, cy)
    lon, lat = ref.pixel_to_lonlat(rpx, rpy)

    # 3. Yaw: map the frame's "up" direction (0,-1) onto the tile and measure angle.
    rpx_up, rpy_up = _apply_H(H, cx, cy - 1.0)
    dx, dy = rpx_up - rpx, rpy_up - rpy            # in reference pixels (y down)
    # Tile is north-up; image row increases downward, so north is -y.
    yaw = math.degrees(math.atan2(dx, -dy))         # 0 = pointing north
    yaw = (yaw + 360.0) % 360.0

    # 4. Scale: how many reference pixels does 1 frame pixel span at the center?
    rpx_r, rpy_r = _apply_H(H, cx + 1.0, cy)
    ref_px_per_frame_px = math.hypot(rpx_r - rpx, rpy_r - rpy)
    mpp_x, _ = ref.meters_per_pixel
    # meters_per_pixel is projected EPSG:3857 metres, inflated by 1/cos(lat)
    # vs true ground metres — correct it so the field really is GROUND m/px.
    ground_m_per_px = ref_px_per_frame_px * mpp_x * math.cos(math.radians(lat))

    return PoseEstimate(
        ok=True, lat=lat, lon=lon, yaw_deg=yaw, ground_m_per_px=ground_m_per_px,
    )


def localize_frame(
    frame_bgr: np.ndarray,
    ref: GeoImage,
    matcher: Matcher,
) -> tuple[PoseEstimate, MatchResult]:
    """Match one frame to the reference tile and estimate its geographic pose."""
    mr = matcher.match(frame_bgr, ref.image)
    if not mr.ok:
        return PoseEstimate(ok=False, n_matches=mr.n_matches), mr
    pose = pose_from_homography(mr.homography, frame_bgr.shape, ref)
    pose.n_inliers = mr.n_inliers
    pose.n_matches = mr.n_matches
    return pose, mr
