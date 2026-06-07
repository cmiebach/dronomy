"""Pluggable georeferenced satellite-imagery providers.

The brief names Google Earth as the primary source; Adrian explicitly sanctioned
open satellite APIs as a fallback. Both implement `ReferenceProvider.fetch()`,
returning a `GeoImage` with an exact pixel<->lat/lon mapping.
"""
from .base import ReferenceProvider, get_provider  # noqa: F401
from .geo import GeoImage  # noqa: F401
from .store import save_reference, load_reference  # noqa: F401
