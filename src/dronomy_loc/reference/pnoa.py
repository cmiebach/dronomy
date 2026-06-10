"""Spanish IGN PNOA provider via the INSPIRE WMS (open data, no API key).

PNOA-MA is the maximum-actuality national orthophoto (~0.10-0.25 m/px) — the
highest-resolution georeferenced source for the Asturias flight area. We request
a GetMap with an explicit EPSG:3857 bounding box, so the returned raster has an
exact pixel<->meter mapping.

Docs: https://www.ign.es/web/ign/portal/ide-area-nodo-ide-ign
Layer: OI.OrthoimageCoverage  (PNOA maximum-actuality orthoimagery)
"""
from __future__ import annotations

import io
import time

import numpy as np
import requests
from PIL import Image

from .base import ReferenceProvider
from .geo import GeoImage, mercator_bbox_around

_DEFAULT_WMS = "https://www.ign.es/wms-inspire/pnoa-ma"
_DEFAULT_LAYER = "OI.OrthoimageCoverage"


class PNOAProvider(ReferenceProvider):
    def __init__(self, cfg=None):
        pnoa = getattr(getattr(cfg, "reference", None), "pnoa", None) if cfg else None
        self.wms_url = getattr(pnoa, "wms_url", _DEFAULT_WMS)
        self.layer = getattr(pnoa, "layer", _DEFAULT_LAYER)
        # Use PNG for the array path; the bbox we pass IS the georeferencing.
        self.image_format = "image/png"
        self.timeout = 60

    def fetch(self, lat: float, lon: float, span_meters: float, pixels: int) -> GeoImage:
        bbox = mercator_bbox_around(lon, lat, span_meters)  # (minx,miny,maxx,maxy) 3857
        params = {
            "SERVICE": "WMS",
            "VERSION": "1.3.0",
            "REQUEST": "GetMap",
            "LAYERS": self.layer,
            "STYLES": "",
            "CRS": "EPSG:3857",
            # WMS 1.3.0 + projected CRS => bbox axis order is minx,miny,maxx,maxy.
            "BBOX": ",".join(f"{v:.6f}" for v in bbox),
            "WIDTH": str(pixels),
            "HEIGHT": str(pixels),
            "FORMAT": self.image_format,
            "TRANSPARENT": "false",
        }
        # The WMS occasionally answers a transient 502/503 (observed live) —
        # retry the same way EsriProvider does before giving up.
        for attempt in range(3):
            resp = requests.get(self.wms_url, params=params, timeout=self.timeout)
            if resp.status_code < 500:
                break
            if attempt < 2:
                time.sleep(1.0 * (1.5 ** attempt))
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if "image" not in ctype:
            # WMS errors come back as XML with a 200 status — surface them clearly.
            raise RuntimeError(f"PNOA WMS did not return an image ({ctype}):\n{resp.text[:500]}")
        img = np.asarray(Image.open(io.BytesIO(resp.content)).convert("RGB"))
        return GeoImage(image=img, bbox=bbox)
