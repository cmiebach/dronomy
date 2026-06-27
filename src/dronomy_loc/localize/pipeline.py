"""Turn a drone↔reference homography into a geographic pose estimate.

Given homography H (drone-frame pixels -> reference-tile pixels) and the reference
`GeoImage` (reference pixels -> lat/lon), we:
  1. project the drone-frame CENTER through H to get its reference pixel,
  2. convert that to (lat, lon)  ->  the ground point the optical axis hits,
  3. derive yaw from how H rotates the frame's up-vector on the (north-up) tile,
  4. derive a scale (ground meters per drone pixel) as a rough altitude proxy.

Step 2 returns the BORESIGHT ground point — where the camera's optical axis
intersects the ground — which equals the drone's nadir only for a perfectly
nadir (straight-down) camera. On an OBLIQUE frame the nadir (the point directly
below the drone, i.e. its true map position) is offset from the boresight by
~`altitude * tan(tilt)`: a 3-5 deg tilt at 80 m is already several metres, and a
30 deg oblique is ~46 m of pure bias. When camera intrinsics are supplied we
recover that offset by decomposing the planar homography into a full camera pose
(rotation + camera centre relative to the ground plane, K [r1 r2 t]) and report
the NADIR instead of the boresight, plus the recovered tilt and altitude.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..matching.base import Matcher, MatchResult
from ..reference.geo import GeoImage, mercator_to_lonlat

if TYPE_CHECKING:  # import-only type; runtime stays duck-typed to avoid a cycle
    from ..framework.schema import CameraIntrinsics


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
    tilt_deg: float | None = None       # camera tilt off nadir (None unless intrinsics given)
    altitude_m: float | None = None     # camera height above the ground plane (geometry-derived)
    tilt_corrected: bool = False        # True when lat/lon is the nadir, not the boresight


@dataclass
class CameraGeometry:
    """Camera pose relative to the local ground plane, recovered from a planar
    homography + intrinsics. The nadir offset is expressed in metres EAST/NORTH
    of the boresight ground point (the frame-centre projection)."""
    east_m: float          # nadir offset from the boresight, +east
    north_m: float         # nadir offset from the boresight, +north
    altitude_m: float      # camera height above the plane
    tilt_deg: float        # angle of the optical axis off straight-down


def _apply_H(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
    v = H @ np.array([x, y, 1.0])
    return v[0] / v[2], v[1] / v[2]


# A camera looking at the ground from above with a tilt below this is treated as
# a trustworthy planar decomposition; beyond it the homography is too grazing for
# the recovered nadir to be reliable, so we fall back to the boresight position.
_MAX_TILT_DEG = 75.0


def decompose_ground_homography(
    G: np.ndarray, K: np.ndarray,
) -> CameraGeometry | None:
    """Decompose a planar homography into the camera pose over the ground plane.

    `G` maps FRAME pixels -> local ground metres (east, north) with the frame
    centre sitting at the metric origin (so the boresight ground point is (0,0)).
    `K` is the 3x3 intrinsics in frame pixels. For a planar scene the
    plane->image map is P = inv(G) = K [r1 r2 t] up to scale; we recover R and the
    translation, then the camera centre C = -R^T t in plane coordinates. C[:2] is
    the nadir offset from the boresight and C[2] the altitude.

    Returns None when G is singular or the recovered pose is physically
    implausible (camera below the plane, NaN, or tilt past `_MAX_TILT_DEG`)."""
    try:
        P = np.linalg.inv(np.asarray(G, dtype=float))
        B = np.linalg.inv(np.asarray(K, dtype=float)) @ P
    except np.linalg.LinAlgError:
        return None
    b1, b2, b3 = B[:, 0], B[:, 1], B[:, 2]
    lam = (np.linalg.norm(b1) + np.linalg.norm(b2)) / 2.0
    if not np.isfinite(lam) or lam <= 0:
        return None
    # The scale sign is ambiguous; both ±lam give a valid rotation. Keep the
    # solution that puts the camera ABOVE the plane (altitude > 0).
    best: CameraGeometry | None = None
    for s in (lam, -lam):
        r1, r2, t = b1 / s, b2 / s, b3 / s
        R0 = np.column_stack([r1, r2, np.cross(r1, r2)])
        # Nearest true rotation (Zhang): SVD-clean R0, fixing a reflection.
        U, _, Vt = np.linalg.svd(R0)
        R = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
        C = -R.T @ t
        if not np.all(np.isfinite(C)) or C[2] <= 0:
            continue
        # Optical axis is +Z_cam; in world coords that is R[2, :]. A nadir camera
        # points straight down (world -Z), so its world-z component is -1.
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -R[2, 2]))))
        if tilt > _MAX_TILT_DEG:
            continue
        if best is None or C[2] > 0:  # first physically valid solution wins
            best = CameraGeometry(float(C[0]), float(C[1]), float(C[2]), tilt)
            break
    return best


def _intrinsics_matrix(intr: CameraIntrinsics, w: int, h: int) -> np.ndarray:
    """3x3 K in frame pixels. Principal point defaults to the frame centre."""
    cx, cy = intr.principal_point if intr.principal_point is not None else (w / 2.0, h / 2.0)
    f = float(intr.focal_px)
    return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])


def pose_from_homography(
    H: np.ndarray, frame_shape, ref: GeoImage,
    intrinsics: CameraIntrinsics | None = None,
) -> PoseEstimate:
    h, w = frame_shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # 1-2. Frame center -> reference pixel -> lat/lon (the boresight ground point).
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
    mpp_x, mpp_y = ref.meters_per_pixel
    cos_lat = math.cos(math.radians(lat))
    # meters_per_pixel is projected EPSG:3857 metres, inflated by 1/cos(lat)
    # vs true ground metres — correct it so the field really is GROUND m/px.
    ground_m_per_px = ref_px_per_frame_px * mpp_x * cos_lat

    pose = PoseEstimate(
        ok=True, lat=lat, lon=lon, yaw_deg=yaw, ground_m_per_px=ground_m_per_px,
    )

    # 5. Oblique-tilt correction: with intrinsics, decompose the homography to put
    # the report on the NADIR (drone's map position) rather than the boresight.
    if intrinsics is not None and getattr(intrinsics, "focal_px", None):
        bx, by = ref.pixel_to_mercator(rpx, rpy)   # boresight in mercator metres
        # Affine: tile px -> local ground metres (east, north), boresight at origin.
        # mercator->ground multiplies by cos(lat) to undo the web-mercator inflation.
        T = np.array([
            [cos_lat * mpp_x, 0.0, cos_lat * (ref.bbox[0] - bx)],
            [0.0, -cos_lat * mpp_y, cos_lat * (ref.bbox[3] - by)],
            [0.0, 0.0, 1.0],
        ])
        G = T @ np.asarray(H, dtype=float)
        geom = decompose_ground_homography(G, _intrinsics_matrix(intrinsics, w, h))
        if geom is not None:
            nx = bx + geom.east_m / cos_lat        # nadir back in mercator metres
            ny = by + geom.north_m / cos_lat
            pose.lon, pose.lat = mercator_to_lonlat(nx, ny)
            pose.tilt_deg = geom.tilt_deg
            pose.altitude_m = geom.altitude_m
            pose.tilt_corrected = True

    return pose


def localize_frame(
    frame_bgr: np.ndarray,
    ref: GeoImage,
    matcher: Matcher,
    intrinsics: CameraIntrinsics | None = None,
) -> tuple[PoseEstimate, MatchResult]:
    """Match one frame to the reference tile and estimate its geographic pose.
    When `intrinsics` is supplied the pose is tilt-corrected to the nadir."""
    mr = matcher.match(frame_bgr, ref.image)
    if not mr.ok:
        return PoseEstimate(ok=False, n_matches=mr.n_matches), mr
    pose = pose_from_homography(mr.homography, frame_bgr.shape, ref, intrinsics)
    pose.n_inliers = mr.n_inliers
    pose.n_matches = mr.n_matches
    return pose, mr
