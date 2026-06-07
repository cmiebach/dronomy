"""Georeferencing core: Web-Mercator (EPSG:3857) <-> WGS84 (lat/lon) math and a
`GeoImage` that maps pixel coordinates to geographic coordinates.

We implement the Web-Mercator transform by hand so the project has zero hard
dependency on pyproj/GDAL. A reference tile fetched with a known bounding box
(in meters, EPSG:3857) gives an exact linear pixel<->meter mapping; combine that
with the mercator<->lat/lon equations below and any pixel resolves to (lat, lon).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

R_EARTH = 6378137.0  # Web-Mercator sphere radius (meters)


# ── WGS84 <-> Web-Mercator scalar transforms ──────────────────────────
def lonlat_to_mercator(lon: float, lat: float) -> tuple[float, float]:
    """(lon, lat) degrees -> (x, y) meters in EPSG:3857."""
    x = math.radians(lon) * R_EARTH
    y = R_EARTH * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))
    return x, y


def mercator_to_lonlat(x: float, y: float) -> tuple[float, float]:
    """(x, y) meters in EPSG:3857 -> (lon, lat) degrees."""
    lon = math.degrees(x / R_EARTH)
    lat = math.degrees(2.0 * math.atan(math.exp(y / R_EARTH)) - math.pi / 2.0)
    return lon, lat


def meters_per_degree_lat(lat: float) -> tuple[float, float]:
    """Approx meters per degree of (lat, lon) at a given latitude — handy for
    quick error reporting in metric units."""
    lat_r = math.radians(lat)
    m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat_r) + 1.175 * math.cos(4 * lat_r)
    m_per_deg_lon = 111412.84 * math.cos(lat_r) - 93.5 * math.cos(3 * lat_r)
    return m_per_deg_lat, m_per_deg_lon


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters between two WGS84 points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R_EARTH * math.asin(math.sqrt(a))


def mercator_bbox_around(lon: float, lat: float, span_m: float) -> tuple[float, float, float, float]:
    """Square EPSG:3857 bbox (minx, miny, maxx, maxy) of side `span_m`, centred
    on (lon, lat). Note: `span_m` is in projected mercator meters, which are
    inflated by 1/cos(lat) relative to true ground meters — fine as a footprint
    request; precise ground scale comes from the GeoImage mapping itself."""
    cx, cy = lonlat_to_mercator(lon, lat)
    half = span_m / 2.0
    return cx - half, cy - half, cx + half, cy + half


@dataclass
class GeoImage:
    """A raster with a known EPSG:3857 bounding box, enabling pixel<->lat/lon.

    image    : HxWx3 (or HxW) numpy array
    bbox     : (minx, miny, maxx, maxy) in EPSG:3857 meters
    Convention: pixel (0,0) is the TOP-LEFT; +x right, +y down. Mercator +y is up,
    so the row axis is flipped relative to mercator y.
    """

    image: "object"  # numpy.ndarray; typed loosely to avoid importing numpy here
    bbox: tuple[float, float, float, float]

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def meters_per_pixel(self) -> tuple[float, float]:
        minx, miny, maxx, maxy = self.bbox
        return (maxx - minx) / self.width, (maxy - miny) / self.height

    def pixel_to_mercator(self, px: float, py: float) -> tuple[float, float]:
        minx, miny, maxx, maxy = self.bbox
        x = minx + (px / self.width) * (maxx - minx)
        y = maxy - (py / self.height) * (maxy - miny)  # row 0 == top == maxy
        return x, y

    def mercator_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        minx, miny, maxx, maxy = self.bbox
        px = (x - minx) / (maxx - minx) * self.width
        py = (maxy - y) / (maxy - miny) * self.height
        return px, py

    def pixel_to_lonlat(self, px: float, py: float) -> tuple[float, float]:
        return mercator_to_lonlat(*self.pixel_to_mercator(px, py))

    def lonlat_to_pixel(self, lon: float, lat: float) -> tuple[float, float]:
        return self.mercator_to_pixel(*lonlat_to_mercator(lon, lat))
