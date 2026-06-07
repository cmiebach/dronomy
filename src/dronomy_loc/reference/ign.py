"""IGN BD ORTHO provider via the Géoplateforme WMS (open data, no API key).

This is the sanctioned open-data fallback and is the most convenient *georeferenced*
source for the French recording location: we request a GetMap with an explicit
EPSG:3857 bounding box, so the returned raster has an exact pixel<->meter mapping.

Docs: https://geoservices.ign.fr/services-geoplateforme-diffusion
Layer: HR.ORTHOIMAGERY.ORTHOPHOTOS  (very-high-resolution orthophotos)
"""
from __future__ import annotations

import io

import numpy as np
import requests
from PIL import Image

from .base import ReferenceProvider
from .geo import GeoImage, mercator_bbox_around

_DEFAULT_WMS = "https://data.geopf.fr/wms-r/wms"
_DEFAULT_LAYER = "HR.ORTHOIMAGERY.ORTHOPHOTOS"


class IGNProvider(ReferenceProvider):
    def __init__(self, cfg=None):
        ign = getattr(getattr(cfg, "reference", None), "ign", None) if cfg else None
        self.wms_url = getattr(ign, "wms_url", _DEFAULT_WMS)
        self.layer = getattr(ign, "layer", _DEFAULT_LAYER)
        # Use PNG/JPEG for the array path; the bbox we pass IS the georeferencing.
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
        }
        resp = requests.get(self.wms_url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if "image" not in ctype:
            # WMS errors come back as XML with a 200 status — surface them clearly.
            raise RuntimeError(f"IGN WMS did not return an image ({ctype}):\n{resp.text[:500]}")
        img = np.asarray(Image.open(io.BytesIO(resp.content)).convert("RGB"))
        return GeoImage(image=img, bbox=bbox)
