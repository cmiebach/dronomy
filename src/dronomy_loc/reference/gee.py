"""Google Earth Engine provider (PRIMARY source per the brief) — STUB.

GEE is the source named in the challenge spec. It requires authentication
(`earthengine authenticate`, a Google account, and a registered cloud project),
so this is left as a documented stub to be fleshed out once auth is set up.

Implementation sketch
---------------------
1. `import ee; ee.Initialize(project=...)`
2. Build an `ee.Geometry.Rectangle` from a mercator/lat-lon bbox centred on the
   target (reuse `geo.mercator_bbox_around` -> convert corners back to lon/lat).
3. Pick a high-res, recent, cloud-free image:
     - For France, Sentinel-2 (`COPERNICUS/S2_SR_HARMONIZED`) sorted by date with
       a cloud filter, or a national/commercial high-res collection if available.
4. `getThumbURL` / `ee.Image.getDownloadURL` with `dimensions=pixels`, `region=bbox`,
   `crs="EPSG:3857"` -> download bytes -> decode to ndarray.
5. Wrap in `GeoImage(image=arr, bbox=bbox_3857)` so pixel<->lat/lon works the same
   way as the IGN provider.

Until then, prefer `provider: ign` in config.yaml (sanctioned open-data fallback).
"""
from __future__ import annotations

from .base import ReferenceProvider
from .geo import GeoImage


class GEEProvider(ReferenceProvider):
    def __init__(self, cfg=None):
        self.cfg = cfg

    def fetch(self, lat: float, lon: float, span_meters: float, pixels: int) -> GeoImage:
        raise NotImplementedError(
            "GEEProvider is a stub. Set up Earth Engine auth and implement the "
            "export per the module docstring, or use provider: 'ign' for now."
        )
