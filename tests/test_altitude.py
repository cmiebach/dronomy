"""Telemetry-free AGL altitude checks - offline, deterministic, no torch/network.

Builds a synthetic north-up GeoImage and feeds hand-built homographies (drone px
-> tile px) so the expected footprint/altitude is known in closed form. The cos(lat)
correction is applied in the expected-value math so it matches the implementation
(GeoImage.meters_per_pixel is projected mercator m/px, inflated by 1/cos(lat))."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronomy_loc.reference.geo import GeoImage, mercator_bbox_around  # noqa: E402
from dronomy_loc.localize.altitude import estimate_altitude  # noqa: E402

LAT, LON = 43.521955, -5.624290  # the recording location
SPAN_M, TILE_PX = 1500.0, 1024
FRAME_W, FRAME_H = 1600, 900     # 16:9 drone frame
HFOV = 84.0


def _ref() -> GeoImage:
    bbox = mercator_bbox_around(LON, LAT, SPAN_M)
    return GeoImage(image=np.zeros((TILE_PX, TILE_PX, 3), np.uint8), bbox=bbox)


def _scale_H(sx: float, sy: float) -> np.ndarray:
    """Frame px -> tile px with per-axis scale, centred on the tile centre."""
    tx = TILE_PX / 2.0 - sx * FRAME_W / 2.0
    ty = TILE_PX / 2.0 - sy * FRAME_H / 2.0
    return np.array([[sx, 0.0, tx], [0.0, sy, ty], [0.0, 0.0, 1.0]])


def test_pure_scale_recovers_expected_agl():
    ref = _ref()
    s = 0.4
    H = _scale_H(s, s)
    est = estimate_altitude(H, (FRAME_H, FRAME_W, 3), ref, hfov_deg=HFOV)

    # Expected footprint width: s*FRAME_W tile px -> ground metres (cos-corrected).
    rcx = s * FRAME_W / 2.0 + (TILE_PX / 2.0 - s * FRAME_W / 2.0)
    rcy = s * FRAME_H / 2.0 + (TILE_PX / 2.0 - s * FRAME_H / 2.0)
    _, lat = ref.pixel_to_lonlat(rcx, rcy)
    mpp_x, _ = ref.meters_per_pixel
    cos_lat = math.cos(math.radians(lat))
    exp_width_m = s * FRAME_W * mpp_x * cos_lat
    exp_agl = (exp_width_m / 2.0) / math.tan(math.radians(HFOV) / 2.0)

    assert not est.degenerate
    assert est.agl_m is not None
    assert math.isclose(est.ground_width_m, exp_width_m, rel_tol=1e-6)
    assert math.isclose(est.agl_m, exp_agl, rel_tol=1e-2)


def test_constant_altitude_invariant_to_translation():
    # Two homographies differing only in translation describe the same height.
    ref = _ref()
    s = 0.4
    H = _scale_H(s, s)
    H2 = H.copy()
    H2[0, 2] += 40.0
    H2[1, 2] -= 25.0
    a1 = estimate_altitude(H, (FRAME_H, FRAME_W, 3), ref).agl_m
    a2 = estimate_altitude(H2, (FRAME_H, FRAME_W, 3), ref).agl_m
    assert a1 is not None and a2 is not None
    assert math.isclose(a1, a2, rel_tol=1e-3)  # only cos(lat) drift across the tile


def test_degenerate_collinear_homography_flagged():
    # Collapse the y axis: the warped quad becomes a line (zero area), no scale.
    ref = _ref()
    H = np.array([[0.4, 0.0, TILE_PX / 2.0 - 0.4 * FRAME_W / 2.0],
                  [0.0, 0.0, TILE_PX / 2.0],
                  [0.0, 0.0, 1.0]])
    est = estimate_altitude(H, (FRAME_H, FRAME_W, 3), ref)
    assert est.degenerate
    assert est.agl_m is None


def test_tilt_ratio_about_one_for_isotropic_scale():
    ref = _ref()
    est = estimate_altitude(_scale_H(0.4, 0.4), (FRAME_H, FRAME_W, 3), ref)
    assert math.isclose(est.tilt_ratio, 1.0, rel_tol=1e-6)


def test_tilt_ratio_off_one_for_anisotropic_scale():
    # y stretched 1.6x more than x -> tilt_ratio = sx/sy = 1/1.6 = 0.625.
    ref = _ref()
    est = estimate_altitude(_scale_H(0.4, 0.4 * 1.6), (FRAME_H, FRAME_W, 3), ref)
    assert not math.isclose(est.tilt_ratio, 1.0, rel_tol=0.2)
    assert math.isclose(est.tilt_ratio, 1.0 / 1.6, rel_tol=1e-6)
