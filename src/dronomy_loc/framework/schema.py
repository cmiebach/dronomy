"""Standardized data contract for the localization framework.

This is the one-size-fits-all schema every dataset maps into, so the SAME engine
(grid-search matching, VO, scoring) runs unchanged on the provided drone video,
UAV-VisLoc, SatLoc, or any future source. A dataset adapter (see `datasets/`)
turns its raw files into a `Scenario`; the runner iterates `Scenario.samples()`
and localizes each `Sample` against `Scenario.reference()`.

The crucial reuse: `Scenario.reference()` returns the SAME `FetchTile` type the
existing localizer already consumes (`localize.search.search_localize` /
`validate_frames`). A dataset supplies a FetchTile (a live provider for the
video, a local georeferenced-tile crop for UAV-VisLoc); the models consume it
identically. That single shared callable is what makes the engine
dataset-agnostic — no model code changes per dataset.

GROUND-TRUTH RULE: `Sample.gt` (a GPSFix) is for scoring only and is NEVER an
input to localization — the system is telemetry-free by design.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from ..data.telemetry import GPSFix
# FetchTile = Callable[[lat, lon, span_m, pixels], GeoImage] — defined once in
# search.py and re-exported here so the contract and the engine never diverge.
from ..localize.search import FetchTile  # noqa: F401


@dataclass
class CameraIntrinsics:
    """Camera model needed to turn a pixel match into metric scale. The single
    load-bearing value is the focal length in pixels (~3713 for the provided
    DJI Mavic 3E wide camera); the rest are optional refinements."""
    focal_px: float
    principal_point: tuple[float, float] | None = None   # (cx, cy); defaults to image centre
    dist_coeffs: tuple[float, ...] | None = None          # (k1,k2,p1,p2,k3); usually negligible here
    hfov_deg: float | None = None                         # horizontal field of view, if known


@dataclass
class Sample:
    """One standardized localization unit: a drone frame plus what is known
    about it. `image_bgr` is an HxWx3 BGR numpy array (typed loosely to keep
    this module import-light, matching `GeoImage`'s convention)."""
    frame_id: int                      # index within the scenario (also the GT join key)
    image_bgr: "object"                # numpy.ndarray, BGR
    t_s: float | None = None           # timestamp in seconds, when known
    gt: GPSFix | None = None           # ground truth (SCORING ONLY — never a model input)
    intrinsics: CameraIntrinsics | None = None
    meta: dict = field(default_factory=dict)   # free-form: heading, altitude, source filename, ...


@dataclass
class Scenario:
    """One localizable unit (a flight, or one region of a multi-region dataset).

    A dataset adapter builds Scenarios by injecting two things:
      * `sample_iter` — a zero-arg factory returning a FRESH iterator of Samples
        (a factory, not an iterator, so the scenario can be replayed); and
      * `fetch_tile`  — the reference accessor (the shared `FetchTile`).
    Everything else is metadata the runner/report use to group and label results.
    """
    name: str
    terrain: str                       # e.g. "forest", "river", "urban", "campus", "mixed", "unknown"
    fetch_tile: FetchTile              # reference() accessor — the shared engine contract
    sample_iter: Callable[[], Iterator[Sample]]
    prior: tuple[float, float] | None = None   # coarse (lat, lon) search prior
    intrinsics: CameraIntrinsics | None = None
    meta: dict = field(default_factory=dict)   # dataset name, source, capture date, region id, ...

    def samples(self) -> Iterator[Sample]:
        """A fresh iterator over this scenario's standardized samples."""
        return self.sample_iter()

    def reference(self) -> FetchTile:
        """The reference-imagery accessor consumed by the localizer unchanged."""
        return self.fetch_tile
