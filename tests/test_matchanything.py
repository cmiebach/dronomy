"""MatchAnything adapter — offline, imcui mocked (real weights live in Docker)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronomy_loc.matching import get_matcher  # noqa: E402
from dronomy_loc.matching import matchanything as ma  # noqa: E402
from dronomy_loc.matching.base import Matcher  # noqa: E402


class _FakeAPI:
    """Stands in for imcui's ImageMatchingAPI: returns a fixed correspondence
    set (a known translation) so we can assert the adapter's plumbing/RANSAC."""
    def __init__(self, pts0, pts1):
        self.pts0, self.pts1 = pts0, pts1
        self.calls = []

    def __call__(self, rgb0, rgb1):
        self.calls.append((rgb0.shape, rgb1.shape, rgb0.dtype))
        return {"mkeypoints0_orig": self.pts0, "mkeypoints1_orig": self.pts1}


def test_factory_returns_matchanything_without_importing_imcui():
    # imcui is NOT installed; constructing the matcher must not import it.
    m = get_matcher("matchanything")
    assert isinstance(m, Matcher)
    assert get_matcher("ma").__class__ is m.__class__


def test_missing_imcui_raises_clear_error(monkeypatch):
    # Force the real loader path; with no imcui installed it must be a friendly
    # ImportError that points at the Docker env, not a bare ModuleNotFoundError.
    m = get_matcher("matchanything")
    with pytest.raises(ImportError, match="imcui"):
        m.match(np.zeros((64, 64, 3), np.uint8), np.zeros((64, 64, 3), np.uint8))


def test_match_plumbs_points_and_runs_ransac(monkeypatch):
    rng = np.random.default_rng(0)
    pts0 = rng.uniform(10, 600, (60, 2)).astype(np.float32)
    pts1 = pts0 + np.array([12.0, -7.0], np.float32)      # pure translation
    fake = _FakeAPI(pts0, pts1)
    monkeypatch.setattr(ma, "_build_api", lambda model, device: (fake, "matchanything_eloftr"))

    m = get_matcher("matchanything")
    res = m.match(np.zeros((640, 640, 3), np.uint8), np.zeros((640, 640, 3), np.uint8))
    assert res.ok and res.n_matches == 60
    assert res.n_inliers >= 50                # a clean translation -> mostly inliers
    assert fake.calls and fake.calls[0][2] == np.dtype("uint8")   # uint8 RGB in


def test_keypoints_rescaled_to_caller_pixels(monkeypatch):
    # Inputs larger than max_long_edge are downscaled for the net; returned
    # homography must be in the ORIGINAL (caller) pixel space.
    pts = np.array([[100, 100], [700, 120], [120, 700], [700, 700],
                    [400, 400]], np.float32)
    fake = _FakeAPI(pts, pts + np.array([5.0, 5.0], np.float32))
    monkeypatch.setattr(ma, "_build_api", lambda model, device: (fake, "x"))
    m = get_matcher("matchanything")
    m.max_long_edge = 416                       # force a 0.5x downscale of 832px input
    res = m.match(np.zeros((832, 832, 3), np.uint8), np.zeros((832, 832, 3), np.uint8))
    # points were divided by the 0.5 scale -> back up near the original 0..832 range
    assert res.src_pts[:, 0].max() > 800


def test_too_few_matches_returns_no_homography(monkeypatch):
    fake = _FakeAPI(np.zeros((2, 2), np.float32), np.zeros((2, 2), np.float32))
    monkeypatch.setattr(ma, "_build_api", lambda model, device: (fake, "x"))
    res = get_matcher("matchanything").match(
        np.zeros((64, 64, 3), np.uint8), np.zeros((64, 64, 3), np.uint8))
    assert not res.ok and res.homography is None
