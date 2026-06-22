"""Telemetry-free above-ground-level (AGL) altitude from the localization homography.

The DJI track carries no usable altitude (every GPS sample is empty), so we never
had a height for the flight. But a successful localization already encodes it: H
maps drone-frame pixels onto reference-tile pixels, and the tile has a known ground
scale, so projecting the frame's four corners through H gives the frame's GROUND
FOOTPRINT in metres -- a camera-model-free measurement. For a nadir (down-looking)
camera the footprint width and the horizontal field of view then fix the altitude:

    footprint_width = 2 * agl * tan(HFOV / 2)
    =>  agl         = (footprint_width / 2) / tan(HFOV / 2)

The footprint is the rigorous part; AGL inherits the FOV assumption (a few percent
from the exact crop). Our telemetry confirms a nadir gimbal, so the width-span vs
height-span ratio should sit near 1; a value far from 1 flags camera tilt or a bad
homography (`tilt_ratio` below). When the warped quad is near-collinear / zero-area
the scale is meaningless -- we flag `degenerate` and return agl_m=None.

GeoImage.meters_per_pixel is PROJECTED EPSG:3857 metres, inflated by 1/cos(lat) vs
true ground metres, so we multiply by cos(lat) exactly as localize/pipeline.py does
for ground_m_per_px.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..reference.geo import GeoImage

# DJI Mavic 3E wide camera horizontal FOV (deg). Configurable via estimate_altitude.
DEFAULT_HFOV_DEG = 84.0


@dataclass
class AltitudeEstimate:
    agl_m: float | None          # above-ground-level altitude, None if degenerate
    ground_width_m: float        # footprint width on the ground (metres)
    ground_height_m: float       # footprint height on the ground (metres)
    gsd_m_per_px: float          # ground sample distance: ground metres per frame px
    degenerate: bool             # True when the warped quad has no usable scale
    tilt_ratio: float            # x-scale / y-scale; ~1 for level nadir


def _apply_H(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
    v = H @ np.array([x, y, 1.0])
    return v[0] / v[2], v[1] / v[2]


def estimate_altitude(
    H: np.ndarray,
    frame_shape,
    ref: GeoImage,
    hfov_deg: float = DEFAULT_HFOV_DEG,
) -> AltitudeEstimate:
    """Recover AGL altitude from H (drone px -> tile px) for a nadir camera.

    Projects the frame's four corners through H, converts the spanned footprint
    to ground metres via the tile scale (corrected by cos(lat)), and derives AGL
    from the footprint width and `hfov_deg`. Flags `degenerate` when the warped
    quad is near-collinear / zero-area (agl_m is then None)."""
    h, w = frame_shape[:2]
    # Corners TL, TR, BR, BL (drone-frame px) -> reference-tile px.
    corners = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    q = np.array([_apply_H(H, x, y) for x, y in corners])  # 4x2 tile px

    # Shoelace area of the warped quad; near-zero => collinear/degenerate.
    x0, y0 = q[:, 0], q[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    area = 0.5 * abs(float(np.sum(x0 * y1 - x1 * y0)))

    # Width = mean of top + bottom edge lengths; height = mean of left + right.
    top = np.linalg.norm(q[1] - q[0])
    bottom = np.linalg.norm(q[2] - q[3])
    left = np.linalg.norm(q[3] - q[0])
    right = np.linalg.norm(q[2] - q[1])
    width_px = 0.5 * (top + bottom)    # tile px spanned across the frame width
    height_px = 0.5 * (left + right)   # tile px spanned across the frame height

    # Degenerate: any frame side maps to ~nothing, or the quad has ~zero area
    # (a frame this size should cover a non-trivial slice of the tile).
    min_px = 1e-6
    degenerate = (
        width_px < min_px
        or height_px < min_px
        or area < (0.5 * width_px * height_px) * 1e-3
    )

    # Ground scale: projected mercator m/px corrected to true ground m/px.
    rcx, rcy = _apply_H(H, w / 2.0, h / 2.0)
    _, lat = ref.pixel_to_lonlat(rcx, rcy)
    mpp_x, mpp_y = ref.meters_per_pixel
    cos_lat = math.cos(math.radians(lat))
    ground_width_m = width_px * mpp_x * cos_lat
    ground_height_m = height_px * mpp_y * cos_lat
    gsd_m_per_px = ground_width_m / w if w else 0.0

    # Per-pixel scale x vs y; ~1 when isotropic (level nadir), regardless of the
    # frame's own aspect. Far from 1 => stretched/tilted homography.
    sx = width_px / w if w else 0.0
    sy = height_px / h if h else 0.0
    tilt_ratio = (sx / sy) if sy > min_px else float("inf")

    half = math.tan(math.radians(hfov_deg) / 2.0)
    agl_m = None if degenerate else (ground_width_m / 2.0) / half

    return AltitudeEstimate(
        agl_m=agl_m,
        ground_width_m=ground_width_m,
        ground_height_m=ground_height_m,
        gsd_m_per_px=gsd_m_per_px,
        degenerate=bool(degenerate),
        tilt_ratio=float(tilt_ratio),
    )
