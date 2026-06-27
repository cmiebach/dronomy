"""Manual anchoring: integrate a relative VO track and place it absolutely from
operator control points. Offline, deterministic (synthetic homographies)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronomy_loc.localize.manual_anchor import (  # noqa: E402
    ManualAnchor, anchor_trajectory, integrate_relative_track, fit_similarity)
from dronomy_loc.localize.odometry import PairwiseLink  # noqa: E402
from dronomy_loc.reference.geo import meters_per_degree_lat  # noqa: E402


def _trans(tx, ty):
    H = np.eye(3); H[0, 2] = tx; H[1, 2] = ty
    return H


def _rot_trans(deg, tx, ty):
    a = math.radians(deg)
    H = np.array([[math.cos(a), -math.sin(a), tx],
                  [math.sin(a), math.cos(a), ty], [0, 0, 1.0]])
    return H


def test_integrate_straight_line():
    links = [PairwiseLink(i, i + 1, _trans(10, 0), 50) for i in range(3)]
    rel = integrate_relative_track(links)
    assert rel[0] == (0.0, 0.0)
    assert abs(rel[3][0] - 30.0) < 1e-9 and abs(rel[3][1]) < 1e-9


def test_integrate_respects_yaw_and_stops_at_break():
    # straight, then a 90 deg turn, then forward -> an L; a None link stops it.
    links = [PairwiseLink(0, 1, _trans(10, 0), 50),
             PairwiseLink(1, 2, _rot_trans(90, 10, 0), 50),
             PairwiseLink(2, 3, _trans(10, 0), 50)]
    rel = integrate_relative_track(links)
    assert abs(rel[3][0] - 20.0) < 1e-6 and abs(rel[3][1] - 10.0) < 1e-6  # turned north
    broken = [PairwiseLink(0, 1, _trans(10, 0), 50),
              PairwiseLink(1, 2, None, 0), PairwiseLink(2, 3, _trans(10, 0), 50)]
    assert set(integrate_relative_track(broken)) == {0, 1}


def test_fit_similarity_recovers_known_transform():
    src = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    a = math.radians(30)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    dst = 2.0 * (R @ src.T).T + np.array([5.0, -3.0])
    R2, s2, t2 = fit_similarity(src, dst)
    assert abs(s2 - 2.0) < 1e-6
    assert np.allclose(s2 * (R2 @ src.T).T + t2, dst, atol=1e-6)


def test_anchor_trajectory_places_full_track():
    links = [PairwiseLink(0, 1, _trans(10, 0), 50),
             PairwiseLink(1, 2, _rot_trans(90, 10, 0), 50),
             PairwiseLink(2, 3, _trans(10, 0), 50)]
    rel = integrate_relative_track(links)               # L-shape, non-collinear
    ref_lat, ref_lon = 43.5220, -5.6243
    m_lat, m_lon = meters_per_degree_lat(ref_lat)

    # Place 3 anchors so geo metres == relative coords (identity similarity):
    # east = x, north = y at scale 1.
    def geo(f):
        x, y = rel[f]
        return ref_lat + y / m_lat, ref_lon + x / m_lon
    anchors = [ManualAnchor(f, *geo(f)) for f in (0, 1, 3)]  # non-collinear

    out = anchor_trajectory(links, anchors)
    assert set(out) == set(rel)
    for f in rel:                                       # every frame recovered ~exactly
        elat, elon = geo(f)
        assert abs(out[f][0] - elat) < 1e-6
        assert abs(out[f][1] - elon) < 1e-6


def test_requires_two_control_points():
    links = [PairwiseLink(0, 1, _trans(10, 0), 50)]
    with pytest.raises(ValueError, match="2 control points"):
        anchor_trajectory(links, [ManualAnchor(0, 43.5, -5.6)])
