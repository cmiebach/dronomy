"""Offline tests for the UAV-VisLoc adapter.

Everything is synthetic and deterministic: a tiny on-disk fixture (two RGB jpgs +
a textured satellite map + a GT CSV + a range CSV) is built in tmp_path, then the
adapter is asked to standardize it into one Scenario. No network, no GPU, no real
imagery. One fixture path carries a non-ASCII component to guard the Windows
np.fromfile / cv2.imdecode IO path.
"""
from __future__ import annotations

import csv

import cv2
import numpy as np
import pytest

from dronomy_loc.datasets.uavvisloc import UAVVisLocDataset
from dronomy_loc.datasets.base import get_dataset
from dronomy_loc.framework.schema import Sample, Scenario
from dronomy_loc.reference.geo import GeoImage, haversine_m

# Two GT points ~tens of metres apart, inside a ~600 m satellite footprint.
PT1 = (43.5220, -5.6250)   # (lat, lon)
PT2 = (43.5219, -5.6240)
# Satellite map geographic extent: a box comfortably enclosing both points
# (~600 m on a side around the midpoint).
MAP_MIN_LON, MAP_MIN_LAT = -5.6290, 43.5190
MAP_MAX_LON, MAP_MAX_LAT = -5.6200, 43.5250


def _write_jpg(path, color):
    """Write a tiny textured RGB jpg via the Windows-safe imencode+tofile path."""
    img = np.zeros((24, 32, 3), dtype=np.uint8)
    img[:] = color
    img[::3, ::3] = (255, 255, 255)            # a little texture
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    buf.tofile(str(path))


def _write_satellite(path):
    """A small textured satellite map (gradient + grid) written as .png."""
    sat = np.zeros((256, 256, 3), dtype=np.uint8)
    xs = np.linspace(0, 255, 256, dtype=np.uint8)
    sat[..., 0] = xs[None, :]                  # horizontal gradient (B in BGR)
    sat[..., 1] = xs[:, None]                  # vertical gradient (G)
    sat[::16, :] = (255, 255, 255)             # grid lines for texture
    sat[:, ::16] = (255, 255, 255)
    ok, buf = cv2.imencode(".png", sat)
    assert ok
    buf.tofile(str(path))


def _build_fixture(root):
    """Create root/'01'/ with drone images, a satellite map, a GT CSV and a
    top-level range CSV. Returns the two drone filenames."""
    region = root / "01"
    drone = region / "drone"
    f1, f2 = "00001.JPG", "00002.JPG"
    _write_jpg(drone / f1, (40, 80, 160))
    _write_jpg(drone / f2, (160, 80, 40))

    sat_name = "satellite01.png"
    _write_satellite(region / sat_name)

    # Per-region GT CSV: deliberately use alias column names ('image','latitude',
    # 'longitude','altitude','yaw') to exercise the tolerant matcher.
    with open(region / "01.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "latitude", "longitude", "altitude", "yaw"])
        w.writerow([f1, PT1[0], PT1[1], 120.0, 33.0])
        w.writerow([f2, PT2[0], PT2[1], 121.5, 47.5])

    # Top-level range CSV: use LT/RB corner column names.
    with open(root / "satellite_coordinates_range.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["mapname", "LT_lat_map", "LT_lon_map", "RB_lat_map", "RB_lon_map"])
        # LT = left-top (min lon, max lat); RB = right-bottom (max lon, min lat).
        w.writerow([sat_name, MAP_MAX_LAT, MAP_MIN_LON, MAP_MIN_LAT, MAP_MAX_LON])

    return f1, f2


def test_scenarios_samples_and_reference(tmp_path):
    # Non-ASCII component in the path to exercise Windows-safe image IO.
    root = tmp_path / "uav-visloc-datos-localizacion"
    f1, f2 = _build_fixture(root)

    ds = UAVVisLocDataset(cfg=None, root=root)
    scenarios = ds.scenarios()

    # One Scenario for region '01'.
    assert len(scenarios) == 1
    sc = scenarios[0]
    assert isinstance(sc, Scenario)
    assert sc.name == "uavvisloc-01"
    assert sc.terrain == "unknown"
    assert sc.intrinsics is None
    assert sc.meta == {"dataset": "uavvisloc", "region": "01", "n_drone": 2}

    # samples(): exactly two Samples with image ndarrays and matching GT.
    samples = list(sc.samples())
    assert len(samples) == 2
    assert all(isinstance(s, Sample) for s in samples)
    for i, (s, fname, pt, head) in enumerate([(samples[0], f1, PT1, 33.0),
                                              (samples[1], f2, PT2, 47.5)]):
        assert isinstance(s.image_bgr, np.ndarray)
        assert s.image_bgr.ndim == 3 and s.image_bgr.shape[2] == 3
        assert s.frame_id == i
        assert s.gt is not None
        assert s.gt.lat == pytest.approx(pt[0])
        assert s.gt.lon == pytest.approx(pt[1])
        assert s.gt.frame == s.frame_id
        assert s.meta["filename"] == fname
        assert s.meta["region"] == "01"
        assert s.meta["heading"] == pytest.approx(head)

    # samples() is replayable (factory, not a spent iterator).
    assert len(list(sc.samples())) == 2

    # prior is the bbox center (lat, lon).
    exp_center = ((MAP_MIN_LAT + MAP_MAX_LAT) / 2.0,
                  (MAP_MIN_LON + MAP_MAX_LON) / 2.0)
    assert sc.prior == pytest.approx(exp_center)

    # reference() returns a FetchTile; a center crop round-trips the center
    # within ~2 m (haversine) at its centre pixel.
    fetch = sc.reference()
    center_lat, center_lon = sc.prior
    tile = fetch(center_lat, center_lon, 80.0, 128)
    assert isinstance(tile, GeoImage)
    assert tile.image.shape[:2] == (128, 128)
    got_lon, got_lat = tile.pixel_to_lonlat(tile.width / 2.0, tile.height / 2.0)
    err = haversine_m(center_lat, center_lon, got_lat, got_lon)
    assert err < 2.0, f"center round-trip off by {err:.3f} m"


def test_get_dataset_routes_to_uavvisloc(tmp_path):
    # get_dataset('uavvisloc') -> UAVVisLocDataset; root falls back to default.
    ds = get_dataset("uavvisloc", cfg=None)
    assert isinstance(ds, UAVVisLocDataset)


def test_missing_root_returns_empty(tmp_path, capsys):
    ds = UAVVisLocDataset(cfg=None, root=tmp_path / "does-not-exist")
    assert ds.scenarios() == []
    assert "dataset root not found" in capsys.readouterr().out


def test_region_missing_range_is_skipped(tmp_path, capsys):
    root = tmp_path / "ds"
    _build_fixture(root)
    # Drop the range CSV -> region has no geo extent and rasterio fallback fails
    # on a plain .png -> region skipped with a note (graceful degradation).
    (root / "satellite_coordinates_range.csv").unlink()
    ds = UAVVisLocDataset(cfg=None, root=root)
    assert ds.scenarios() == []
    out = capsys.readouterr().out
    assert "no geo extent" in out
