"""Esri World Imagery provider via the ArcGIS REST export endpoint (no API key).

Global sub-meter satellite/aerial mosaic — the most reliable keyless source for
any flight area. We request an export with an explicit EPSG:3857 bounding box,
so the returned raster has an exact pixel<->meter mapping.

Docs: https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer
"""
from __future__ import annotations

import io
import time

import numpy as np
import requests
from PIL import Image

from .base import ReferenceProvider
from .geo import GeoImage, mercator_bbox_around

_DEFAULT_EXPORT = (
    "https://services.arcgisonline.com/arcgis/rest/services/"
    "World_Imagery/MapServer/export"
)
_MAX_ATTEMPTS = 3       # the server sporadically 500s on some bbox/size combos
_BACKOFF_S = 1.0        # grows 1.5x per retry


class EsriProvider(ReferenceProvider):
    def __init__(self, cfg=None):
        esri = getattr(getattr(cfg, "reference", None), "esri", None) if cfg else None
        self.export_url = getattr(esri, "export_url", _DEFAULT_EXPORT)
        self.timeout = 60

    def fetch(self, lat: float, lon: float, span_meters: float, pixels: int) -> GeoImage:
        bbox = mercator_bbox_around(lon, lat, span_meters)  # (minx,miny,maxx,maxy) 3857
        params = {
            "bbox": ",".join(f"{v:.6f}" for v in bbox),
            "bboxSR": "3857",
            "imageSR": "3857",
            "size": f"{pixels},{pixels}",
            "format": "png",
            "f": "image",
        }
        backoff = _BACKOFF_S
        for attempt in range(_MAX_ATTEMPTS):
            resp = requests.get(self.export_url, params=params, timeout=self.timeout)
            if resp.status_code < 500 or attempt == _MAX_ATTEMPTS - 1:
                break
            time.sleep(backoff)
            backoff *= 1.5
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if "image" not in ctype:
            # Esri reports errors as JSON with a 200 status — surface them clearly.
            raise RuntimeError(f"Esri export did not return an image ({ctype}):\n{resp.text[:500]}")
        img = np.asarray(Image.open(io.BytesIO(resp.content)).convert("RGB"))
        return GeoImage(image=img, bbox=bbox)
