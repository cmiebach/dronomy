"""Model registry, per-scene search scaling, and the localize() wrapper.
Offline: matchers are constructed (no torch/imcui import at construction) and
search_localize is monkeypatched so no real matching/network runs."""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronomy_loc.models import base as mb  # noqa: E402
from dronomy_loc.models import get_model, search_for_altitude, SceneSearch  # noqa: E402
from dronomy_loc.framework.schema import Sample  # noqa: E402
from dronomy_loc.data.telemetry import GPSFix  # noqa: E402


def test_registry_resolves_known_models():
    assert get_model("sift").matcher.__class__.__name__ == "ClassicalMatcher"
    assert get_model("loftr").matcher.__class__.__name__ == "DeepMatcher"
    roma = get_model("roma")
    assert roma.matcher.__class__.__name__ == "MatchAnythingMatcher"
    assert roma.matcher.model == "roma"
    assert get_model("eloftr").matcher.model == "eloftr"
    with pytest.raises(ValueError):
        get_model("nope")


def test_search_scales_with_altitude():
    low = search_for_altitude(50.0, hfov_deg=70.0)    # provided flight
    high = search_for_altitude(466.0, hfov_deg=70.0)  # UAV-VisLoc region 03
    assert low.radius_m < 100 < high.radius_m         # ~70 m vs ~650 m footprint
    assert max(high.scales_m) > max(low.scales_m) * 5
    # No FOV/focal known -> fall back to the safe 50 m defaults.
    assert search_for_altitude(50.0).radius_m == SceneSearch().radius_m


def test_localize_wraps_search_and_scores_without_using_gt(monkeypatch):
    captured = {}

    def fake_search(frame_bgr, prior_lat, prior_lon, matcher, fetch_tile,
                    *, search_radius_m, grid_step_m, scales_m, pixels,
                    min_inliers_lock):
        captured["prior"] = (prior_lat, prior_lon)
        captured["frame_is_sample_image"] = frame_bgr is SENTINEL_IMG
        captured["radius"] = search_radius_m
        pose = types.SimpleNamespace(lat=43.52201, lon=-5.62430, yaw_deg=12.0)
        best = types.SimpleNamespace(pose=pose, n_inliers=77)
        return types.SimpleNamespace(locked=True, best=best)

    monkeypatch.setattr(mb, "search_localize", fake_search)

    SENTINEL_IMG = np.zeros((8, 8, 3), np.uint8)
    gt = GPSFix(frame=42, t_s=1.4, lat=43.52200, lon=-5.62430, alt_m=50.0)
    sample = Sample(frame_id=42, image_bgr=SENTINEL_IMG, t_s=1.4, gt=gt)
    model = mb.LocalizationModel(name="loftr", matcher=object(),
                                 search=SceneSearch(radius_m=99.0))

    fs = model.localize(sample, fetch_tile=lambda *a, **k: None, prior=(43.5219, -5.6243))

    # Wrapping is correct: search got the PRIOR (not GT) and the sample image.
    assert captured["prior"] == (43.5219, -5.6243)
    assert captured["frame_is_sample_image"] is True
    assert captured["radius"] == 99.0
    # Scoring is correct: estimate carried through, error computed vs GT (~1 m).
    assert fs.frame == 42 and fs.locked and fs.n_inliers == 77
    assert fs.est_lat == 43.52201
    assert fs.err_m is not None and fs.err_m < 5.0
    assert fs.yaw_deg == 12.0


def test_localize_handles_no_lock(monkeypatch):
    monkeypatch.setattr(mb, "search_localize",
                        lambda *a, **k: types.SimpleNamespace(locked=False, best=None))
    gt = GPSFix(frame=1, t_s=None, lat=43.5, lon=-5.6, alt_m=None)
    sample = Sample(frame_id=1, image_bgr=np.zeros((8, 8, 3), np.uint8), gt=gt)
    fs = mb.LocalizationModel("sift", object()).localize(
        sample, fetch_tile=lambda *a, **k: None, prior=(43.5, -5.6))
    assert not fs.locked and fs.est_lat is None and fs.err_m is None and fs.n_inliers == 0
