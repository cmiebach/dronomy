"""Per-frame imagery-source selection — offline, deterministic (search mocked)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronomy_loc.localize import multisource  # noqa: E402
from dronomy_loc.data.telemetry import GPSFix  # noqa: E402


def _res(locked, inliers, lat=43.5, lon=-5.6):
    """Fake SearchResult with the attributes multisource reads."""
    best = SimpleNamespace(n_inliers=inliers,
                           pose=SimpleNamespace(lat=lat, lon=lon, yaw_deg=0.0)) if inliers else None
    return SimpleNamespace(locked=locked, best=best)


def _patch(monkeypatch, table):
    """table: fetch-sentinel -> SearchResult. Mocks search_localize by provider."""
    def fake(frame, plat, plon, matcher, fetch, **kw):
        return table[fetch]
    monkeypatch.setattr(multisource, "search_localize", fake)


def test_picks_higher_confidence_provider(monkeypatch):
    A, B = object(), object()
    _patch(monkeypatch, {A: _res(True, 30), B: _res(True, 50)})
    res, name = multisource.localize_multisource(None, 43.5, -5.6, None, {"pnoa": A, "esri": B})
    assert name == "esri" and res.best.n_inliers == 50


def test_falls_back_to_the_only_locking_provider(monkeypatch):
    A, B = object(), object()
    _patch(monkeypatch, {A: _res(True, 22), B: _res(False, 0)})
    res, name = multisource.localize_multisource(None, 43.5, -5.6, None, {"pnoa": A, "esri": B})
    assert name == "pnoa"


def test_none_when_no_provider_locks(monkeypatch):
    A, B = object(), object()
    _patch(monkeypatch, {A: _res(False, 0), B: _res(False, 0)})
    res, name = multisource.localize_multisource(None, 43.5, -5.6, None, {"pnoa": A, "esri": B})
    assert res is None and name is None


def test_validate_multisource_union_coverage_and_choices(monkeypatch):
    # frame 0: only PNOA locks; frame 1: only Esri locks; frame 2: neither.
    A, B = object(), object()
    providers = {"pnoa": A, "esri": B}

    def fake(frame, plat, plon, matcher, fetch, **kw):
        # use a frame-id smuggled via the image arg to vary the result
        fid = frame
        if fid == 0:
            return _res(True, 25) if fetch is A else _res(False, 0)
        if fid == 1:
            return _res(True, 40, lat=43.6) if fetch is B else _res(False, 0)
        return _res(False, 0)
    monkeypatch.setattr(multisource, "search_localize", fake)

    track = [GPSFix(frame=i, t_s=None, lat=43.5, lon=-5.6, alt_m=None) for i in range(3)]
    frames = {0: 0, 1: 1, 2: 2}            # image == frame id (the mock reads it)
    summary, choices = multisource.validate_multisource(
        frames, track, 43.5, -5.6, None, providers, min_inliers_lock=20)

    assert summary.n == 3
    assert summary.n_locked == 2           # union: PNOA(0) + Esri(1)
    by_frame = {c.frame: c.provider for c in choices}
    assert by_frame == {0: "pnoa", 1: "esri", 2: None}


def test_empty_frames(monkeypatch):
    monkeypatch.setattr(multisource, "search_localize", lambda *a, **k: _res(False, 0))
    summary, choices = multisource.validate_multisource(
        {}, [], 43.5, -5.6, None, {"pnoa": object()})
    assert summary.n == 0 and summary.median_err_m is None and choices == []
