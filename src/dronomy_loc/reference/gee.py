"""Google Earth Engine provider via the **map-tiles API**.

Builds a cloud-filtered RGB composite (Sentinel-2 by default) over the target
area, asks Earth Engine for an XYZ *tile endpoint* (`image.getMapId` ->
`tile_fetcher.url_format`), then fetches the Web-Mercator tiles that cover our
bounding box, mosaics them, and crops to the exact bbox. The result is a
`GeoImage` whose `bbox` (EPSG:3857) IS its georeferencing — identical contract
to the IGN provider, so everything downstream (pixel<->lat/lon) is unchanged.

Auth (one-time): `earthengine authenticate`, and a Google Cloud project with the
Earth Engine API enabled. Set the project in config (`reference.gee.project`) or
the `EE_PROJECT` env var.

Resolution caveat: Earth Engine's free global collections (Sentinel-2) are
~10 m/px. We can render tiles at any zoom, but the real detail is ~10 m — coarse
relative to a low-altitude drone frame. If matching is weak, a sub-meter source
(Esri World Imagery / Google Maps Platform Map Tiles) is the higher-res
alternative.
"""
from __future__ import annotations

import io
import math

import numpy as np
import requests
from PIL import Image

from .base import ReferenceProvider
from .geo import GeoImage, mercator_bbox_around, mercator_to_lonlat, R_EARTH

# Web-Mercator world extent (meters). Half = pi*R; the full square is 2*Half.
_HALF = math.pi * R_EARTH
_WORLD = 2.0 * _HALF
_TILE = 256  # standard slippy-map tile size (px)


class GEEProvider(ReferenceProvider):
    def __init__(self, cfg=None):
        gee = getattr(getattr(cfg, "reference", None), "gee", None) if cfg else None
        self.collection = getattr(gee, "collection", "COPERNICUS/S2_SR_HARMONIZED")
        self.start_date = getattr(gee, "start_date", "2023-01-01")
        self.end_date = getattr(gee, "end_date", "2024-12-31")
        self.cloud_prop = getattr(gee, "cloud_prop", "CLOUDY_PIXEL_PERCENTAGE")
        self.max_cloud_pct = float(getattr(gee, "max_cloud_pct", 10))
        self.rgb_bands = list(getattr(gee, "rgb_bands", ["B4", "B3", "B2"]))
        self.vis_min = float(getattr(gee, "vis_min", 0))
        self.vis_max = float(getattr(gee, "vis_max", 3000))
        self.max_zoom = int(getattr(gee, "max_zoom", 17))
        self.project = getattr(gee, "project", "") or None
        self.timeout = 60

    # ── Earth Engine: build a tile endpoint for an RGB composite ──────────
    def _tile_url_format(self, bbox: tuple) -> str:
        try:
            import ee
        except ImportError as exc:  # pragma: no cover - dep guard
            raise RuntimeError(
                "earthengine-api is not installed. `pip install earthengine-api`."
            ) from exc

        import os
        project = self.project or os.getenv("EE_PROJECT")
        # Two auth paths:
        #  * Interactive (local): `earthengine authenticate` stores user creds.
        #  * Headless (Docker/VPS): a service account + JSON key via env vars.
        sa = os.getenv("EE_SERVICE_ACCOUNT")
        sa_key = os.getenv("EE_SERVICE_ACCOUNT_KEY")
        try:
            if sa and sa_key:
                creds = ee.ServiceAccountCredentials(sa, sa_key)
                ee.Initialize(creds, project=project)
            else:
                ee.Initialize(project=project)
        except Exception as exc:  # noqa: BLE001 - EE raises many auth/init types
            raise RuntimeError(
                "Earth Engine not initialised. Run `earthengine authenticate` and "
                "set a project via reference.gee.project (config) or EE_PROJECT "
                f"(env). Underlying error: {type(exc).__name__}: {exc}"
            ) from exc

        minx, miny, maxx, maxy = bbox
        min_lon, min_lat = mercator_to_lonlat(minx, miny)
        max_lon, max_lat = mercator_to_lonlat(maxx, maxy)
        region = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

        col = (ee.ImageCollection(self.collection)
               .filterBounds(region)
               .filterDate(self.start_date, self.end_date))
        if self.cloud_prop:
            col = col.filter(ee.Filter.lt(self.cloud_prop, self.max_cloud_pct))
        # Median composite is robust to residual cloud/shadow gaps.
        image = col.median()
        vis = {"bands": self.rgb_bands, "min": self.vis_min, "max": self.vis_max}
        mapid = image.getMapId(vis)
        return mapid["tile_fetcher"].url_format

    # ── XYZ tile math (Web-Mercator slippy tiles) ─────────────────────────
    def _zoom_for(self, span_m: float, pixels: int) -> int:
        """Smallest zoom whose tile resolution is at least as fine as requested."""
        target_mpp = span_m / pixels  # mercator m/px we want
        z = math.ceil(math.log2(_WORLD / (_TILE * target_mpp)))
        return max(0, min(z, self.max_zoom))

    def _merc_to_global_px(self, x: float, y: float, z: int) -> tuple:
        """EPSG:3857 meters -> global pixel coords at zoom z (origin top-left)."""
        scale = _TILE * (2 ** z)
        gx = (x + _HALF) / _WORLD * scale
        gy = (_HALF - y) / _WORLD * scale  # mercator y is up; pixel y is down
        return gx, gy

    def fetch(self, lat: float, lon: float, span_meters: float, pixels: int) -> GeoImage:
        bbox = mercator_bbox_around(lon, lat, span_meters)  # (minx,miny,maxx,maxy) 3857
        minx, miny, maxx, maxy = bbox
        z = self._zoom_for(span_meters, pixels)
        url_format = self._tile_url_format(bbox)

        # Global-pixel crop box at zoom z (left<right, top<bottom).
        left, top = self._merc_to_global_px(minx, maxy, z)
        right, bottom = self._merc_to_global_px(maxx, miny, z)

        n_tiles = 2 ** z
        tx0, tx1 = math.floor(left / _TILE), math.floor((right - 1e-6) / _TILE)
        ty0, ty1 = math.floor(top / _TILE), math.floor((bottom - 1e-6) / _TILE)

        cols, rows = (tx1 - tx0 + 1), (ty1 - ty0 + 1)
        mosaic = np.zeros((rows * _TILE, cols * _TILE, 3), dtype=np.uint8)
        with requests.Session() as sess:
            for tx in range(tx0, tx1 + 1):
                for ty in range(ty0, ty1 + 1):
                    url = url_format.format(z=z, x=tx % n_tiles, y=ty)
                    resp = sess.get(url, timeout=self.timeout)
                    resp.raise_for_status()
                    tile = np.asarray(Image.open(io.BytesIO(resp.content)).convert("RGB"))
                    oy, ox = (ty - ty0) * _TILE, (tx - tx0) * _TILE
                    mosaic[oy:oy + _TILE, ox:ox + _TILE] = tile

        # Crop the mosaic to the exact bbox, then resize to the requested raster.
        cl, ct = int(round(left - tx0 * _TILE)), int(round(top - ty0 * _TILE))
        cr, cb = int(round(right - tx0 * _TILE)), int(round(bottom - ty0 * _TILE))
        crop = mosaic[ct:cb, cl:cr]
        if crop.shape[0] != pixels or crop.shape[1] != pixels:
            import cv2
            crop = cv2.resize(crop, (pixels, pixels), interpolation=cv2.INTER_AREA)
        return GeoImage(image=crop, bbox=bbox)
