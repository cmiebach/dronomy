"""Runner (IoC spine): benchmarks models on a scenario, scales search per scene,
selects the best model. Offline: search_localize is mocked, synthetic scenario."""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronomy_loc.framework import runner as R  # noqa: E402
from dronomy_loc.models import base as mb  # noqa: E402
from dronomy_loc.models import LocalizationModel  # noqa: E402
from dronomy_loc.framework.schema import Sample, Scenario, CameraIntrinsics  # noqa: E402
from dronomy_loc.data.telemetry import GPSFix  # noqa: E402

BASE_LAT, BASE_LON = 43.5220, -5.6243
_D = 1.0 / 111_320.0
_CAP = {}


def _fake_search(frame_bgr, plat, plon, matcher, fetch_tile, *, search_radius_m,
                 grid_step_m, scales_m, pixels, min_inliers_lock):
    """Pose lands `matcher.q` metres north of BASE (so per-model error == q).
    Records the search radius so the test can assert per-scene scaling."""
    _CAP["radius"] = search_radius_m
    pose = types.SimpleNamespace(lat=BASE_LAT + matcher.q * _D, lon=BASE_LON,
                                 yaw_deg=0.0)
    return types.SimpleNamespace(locked=True,
                                 best=types.SimpleNamespace(pose=pose, n_inliers=50))


def _scenario():
    img = np.zeros((100, 200, 3), np.uint8)  # width 200
    def gt(f):
        return GPSFix(frame=f, t_s=float(f), lat=BASE_LAT, lon=BASE_LON, alt_m=50.0)
    return Scenario(
        name="synthA", terrain="forest",
        fetch_tile=lambda *a, **k: None,
        sample_iter=lambda: iter([Sample(frame_id=f, image_bgr=img, gt=gt(f))
                                  for f in range(2)]),
        prior=(BASE_LAT, BASE_LON),
        intrinsics=CameraIntrinsics(focal_px=3713.0, hfov_deg=70.0),
        meta={"dataset": "video", "altitude_m": 50.0},
    )


def test_run_scenario_benchmarks_and_selects_best(monkeypatch):
    monkeypatch.setattr(mb, "search_localize", _fake_search)
    models = {
        "good": LocalizationModel("good", types.SimpleNamespace(q=1.0)),  # ~1 m
        "bad": LocalizationModel("bad", types.SimpleNamespace(q=9.0)),    # ~9 m
    }
    res = R.run_scenario(_scenario(), models, select_metric="recall_5m")

    assert res.n_samples == 2 and res.dataset == "video" and res.terrain == "forest"
    assert set(res.per_model) == {"good", "bad"}
    assert res.per_model["good"].recall_5m == 1.0   # both frames within 5 m
    assert res.per_model["bad"].recall_5m == 0.0    # ~9 m, misses
    assert res.best_model == "good"


def test_runner_scales_search_per_scene(monkeypatch):
    monkeypatch.setattr(mb, "search_localize", _fake_search)
    R.run_scenario(_scenario(), {"m": LocalizationModel("m", types.SimpleNamespace(q=1.0))})
    # alt 50 m, hfov 70 deg -> footprint ~70 m -> radius ~70, NOT the 120 default.
    assert 50.0 < _CAP["radius"] < 100.0


def test_framework_defaults_from_cfg():
    cfg = types.SimpleNamespace(framework=types.SimpleNamespace(
        models=["loftr"], select_metric="median_err_m", max_samples=5))
    assert R._framework_default(cfg, "models", ["sift"]) == ["loftr"]
    assert R._framework_default(cfg, "select_metric", "recall_5m") == "median_err_m"
    assert R._framework_default(cfg, "max_samples", None) == 5
    # No cfg / no framework block -> fallbacks.
    assert R._framework_default(None, "models", ["sift"]) == ["sift"]
    assert R._framework_default(types.SimpleNamespace(), "models", ["sift"]) == ["sift"]


def test_run_scenario_requires_prior():
    sc = _scenario()
    sc.prior = None
    with pytest.raises(ValueError, match="prior"):
        R.run_scenario(sc, {"m": LocalizationModel("m", types.SimpleNamespace(q=1.0))})
