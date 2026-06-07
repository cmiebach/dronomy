"""Reference-provider interface + factory."""
from __future__ import annotations

import abc

from .geo import GeoImage


class ReferenceProvider(abc.ABC):
    """Fetches a georeferenced satellite tile centred on a lat/lon."""

    @abc.abstractmethod
    def fetch(
        self,
        lat: float,
        lon: float,
        span_meters: float,
        pixels: int,
    ) -> GeoImage:
        """Return a `GeoImage` covering a `span_meters` square centred on
        (lat, lon), rasterised at `pixels` x `pixels`."""
        raise NotImplementedError


def get_provider(name: str, cfg=None) -> ReferenceProvider:
    """Factory: 'ign' | 'gee'. `cfg` is the loaded config namespace (optional)."""
    name = name.lower()
    if name == "ign":
        from .ign import IGNProvider
        return IGNProvider(cfg)
    if name in ("gee", "earthengine", "googleearth"):
        from .gee import GEEProvider
        return GEEProvider(cfg)
    raise ValueError(f"Unknown reference provider: {name!r} (expected 'ign' or 'gee')")
