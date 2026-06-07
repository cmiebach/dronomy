"""Geo math sanity checks — these run without any heavy deps (torch, network)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronomy_loc.reference.geo import (  # noqa: E402
    GeoImage, lonlat_to_mercator, mercator_to_lonlat, mercator_bbox_around, haversine_m,
)

LAT, LON = 43.521955, 5.624290  # the recording location


def test_mercator_roundtrip():
    x, y = lonlat_to_mercator(LON, LAT)
    lon2, lat2 = mercator_to_lonlat(x, y)
    assert lon2 == pytest.approx(LON, abs=1e-9)
    assert lat2 == pytest.approx(LAT, abs=1e-9)


def test_geoimage_center_maps_back():
    span, px = 1500.0, 1024
    bbox = mercator_bbox_around(LON, LAT, span)
    geo = GeoImage(image=np.zeros((px, px, 3), np.uint8), bbox=bbox)
    clon, clat = geo.pixel_to_lonlat(px / 2, px / 2)
    # Center pixel should map back to the requested point within ~1 m.
    assert haversine_m(LAT, LON, clat, clon) < 1.0


def test_pixel_lonlat_roundtrip():
    bbox = mercator_bbox_around(LON, LAT, 1500.0)
    geo = GeoImage(image=np.zeros((512, 512, 3), np.uint8), bbox=bbox)
    for px, py in [(0, 0), (511, 0), (256, 300), (511, 511)]:
        lon, lat = geo.pixel_to_lonlat(px, py)
        bx, by = geo.lonlat_to_pixel(lon, lat)
        assert bx == pytest.approx(px, abs=1e-3)
        assert by == pytest.approx(py, abs=1e-3)


def test_corner_ordering():
    """Top-left pixel must be north-west of the bottom-right pixel."""
    bbox = mercator_bbox_around(LON, LAT, 1000.0)
    geo = GeoImage(image=np.zeros((256, 256, 3), np.uint8), bbox=bbox)
    tl_lon, tl_lat = geo.pixel_to_lonlat(0, 0)
    br_lon, br_lat = geo.pixel_to_lonlat(256, 256)
    assert tl_lat > br_lat  # north is up
    assert tl_lon < br_lon  # west is left
