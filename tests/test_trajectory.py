"""Trajectory shape-precision scoring — offline, deterministic (no plotting)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronomy_loc.localize.trajectory import (  # noqa: E402
    align_se2, lonlat_to_local_m, score_trajectory,
)


def test_align_recovers_known_rotation_translation():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(40, 2)) * 50.0
    th = math.radians(20.0)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    dst = src @ R.T + np.array([10.0, -5.0])     # pure rigid motion
    a = align_se2(src, dst)
    assert a.rot_deg == pytest.approx(20.0, abs=1e-6)
    assert np.allclose(a.apply(src), dst, atol=1e-9)   # exact recovery


def test_alignment_does_not_absorb_scale():
    # A path twice the size must NOT align to zero error (no scale in SE(2)).
    rng = np.random.default_rng(1)
    src = rng.normal(size=(30, 2)) * 20.0
    dst = src * 2.0                                # pure scale
    a = align_se2(src, dst)
    res = np.linalg.norm(a.apply(src) - dst, axis=1)
    assert res.max() > 5.0                         # scale error survives alignment


def test_constant_offset_is_zero_after_alignment():
    # A trajectory shifted by a constant bias has perfect SHAPE -> aligned ~0.
    lat0, lon0 = 43.5219, -5.6243
    t = np.linspace(0, 1, 50)
    gt_lat = lat0 + t * 0.001
    gt_lon = lon0 + np.sin(t * 6) * 0.0005
    est_lat = gt_lat + 0.0002        # ~22 m north constant bias
    est_lon = gt_lon + 0.0
    m = score_trajectory(est_lat, est_lon, gt_lat, gt_lon)
    assert m.ate_raw_m > 15.0                      # raw punishes the offset
    assert m.ate_aligned_m < 0.5                   # shape is identical
    assert m.path_len_ratio == pytest.approx(1.0, abs=1e-3)


def test_local_metric_scale_is_sane():
    # 0.001 deg latitude ~ 111 m at this latitude; east axis shorter (cos lat).
    pts = lonlat_to_local_m([43.5219, 43.5229], [-5.6243, -5.6243], 43.5219, -5.6243)
    assert pts[1, 1] == pytest.approx(111.1, abs=2.0)   # north metres for 0.001 deg
    assert abs(pts[0, 0]) < 1e-6                          # same lon -> east 0


def test_path_length_ratio_detects_wrong_dimensions():
    lat0, lon0 = 43.5219, -5.6243
    t = np.linspace(0, 1, 40)
    gt_lat, gt_lon = lat0 + t * 0.001, lon0 + t * 0.0
    est_lat, est_lon = lat0 + t * 0.0005, lon0 + t * 0.0   # half as long
    m = score_trajectory(est_lat, est_lon, gt_lat, gt_lon)
    assert m.path_len_ratio == pytest.approx(0.5, abs=0.02)
