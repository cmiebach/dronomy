"""Grid-of-centres × multi-scale tile search around a coarse prior.

A drone frame's ground footprint on this footage is only ~50-150 m, but a single
1.5 km reference tile leaves a brutal scale gap, and off-centre repetitive
structures pulled the homography into 80-90 m biases (teammate-measured).
Searching a GRID of candidate tile centres around the coarse prior, crossed with
MULTIPLE tile spans (unknown altitude => unknown scale), and keeping the
candidate with the most RANSAC inliers fixed this: mean error 125.8 m -> 56.8 m.

The inlier count doubles as the confidence gate: on this footage ~4-9 inliers is
the noise floor of repeated texture, while >= 20 is a trustworthy lock
(teammate-calibrated). That threshold is exposed as `min_inliers_lock`.

That absolute gate is calibrated for SPARSE matchers (SIFT/LoFTR). A DENSE
matcher (RoMA / MatchAnything) returns 80-400 "inliers" on essentially every
tile — including the wrong ones — so a flat `>= 20` would lock confidently onto
garbage. The fix is a RELATIVE-margin gate: require the winning peak to
dominate the best SPATIALLY-DISTINCT alternative hypothesis by a ratio
(`lock_margin_ratio`). Adjacent grid cells / scales of the *same* true location
are not alternatives — they are the same peak — so the runner-up is taken only
from candidates at least `margin_separation_m` away from the winner. The gate
defaults to inert (ratio 1.0) so sparse-matcher behaviour is unchanged; callers
running RoMA pass a ratio > 1 (~1.3-1.6 is the recommended starting band).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..matching.base import Matcher
from ..reference.geo import GeoImage, lonlat_to_mercator, mercator_to_lonlat
from .pipeline import PoseEstimate, localize_frame

# (lat, lon, span_m, pixels) -> GeoImage. A provider.fetch, a TileCache, or a test stub.
FetchTile = Callable[[float, float, float, int], GeoImage]


def grid_centers(
    prior_lat: float, prior_lon: float, radius_m: float, step_m: float,
) -> list[tuple[float, float]]:
    """Square grid of candidate tile centres covering ±`radius_m` around the prior,
    spaced `step_m`. Offsets are laid out in projected mercator meters (the same
    space tile bboxes live in) and converted back to lat/lon. Degenerate inputs
    (radius 0, or a step too coarse to fit a neighbour) yield just the prior,
    which is always included exactly."""
    if radius_m <= 0 or step_m <= 0 or step_m >= 2 * radius_m:
        return [(prior_lat, prior_lon)]
    cx, cy = lonlat_to_mercator(prior_lon, prior_lat)
    n = int((radius_m + 1e-9) // step_m)
    offsets = [k * step_m for k in range(-n, n + 1)]
    centers: list[tuple[float, float]] = []
    for dy in offsets:
        for dx in offsets:
            if dx == 0.0 and dy == 0.0:
                centers.append((prior_lat, prior_lon))  # exact: no roundtrip error
            else:
                lon, lat = mercator_to_lonlat(cx + dx, cy + dy)
                centers.append((lat, lon))
    return centers


@dataclass
class Candidate:
    """One (centre, span) cell of the search. `pose` is None when matching (or
    the tile fetch itself) failed."""
    lat: float
    lon: float
    span_m: float
    n_inliers: int = 0
    n_matches: int = 0
    pose: PoseEstimate | None = None


@dataclass
class SearchResult:
    locked: bool                 # passes both the absolute and relative-margin gates
    best: Candidate | None
    candidates: list[Candidate]
    runner_up: Candidate | None = None   # best SPATIALLY-DISTINCT alternative to `best`
    margin_ratio: float | None = None    # best.n_inliers / runner_up.n_inliers (None if no rival)


def _separation_m(a: Candidate, b: Candidate) -> float:
    """Projected (mercator) ground distance between two candidate tile centres."""
    ax, ay = lonlat_to_mercator(a.lon, a.lat)
    bx, by = lonlat_to_mercator(b.lon, b.lat)
    return math.hypot(ax - bx, ay - by)


def _runner_up(best: Candidate, candidates: list[Candidate],
               separation_m: float) -> Candidate | None:
    """The strongest candidate that represents a DIFFERENT location hypothesis
    than `best`: a valid pose at least `separation_m` away. Adjacent cells/scales
    of the same peak are excluded so the margin gate compares real rivals, not a
    peak against its own shoulder."""
    rival: Candidate | None = None
    for c in candidates:
        if c is best or c.pose is None:
            continue
        if _separation_m(best, c) < separation_m:
            continue
        if rival is None or c.n_inliers > rival.n_inliers:
            rival = c
    return rival


def search_localize(
    frame_bgr: np.ndarray,
    prior_lat: float,
    prior_lon: float,
    matcher: Matcher,
    fetch_tile: FetchTile,
    *,
    search_radius_m: float = 120.0,
    grid_step_m: float = 60.0,
    scales_m: tuple[float, ...] = (50.0, 80.0, 110.0, 140.0),
    pixels: int = 640,
    min_inliers_lock: int = 20,
    lock_margin_ratio: float = 1.0,
    margin_separation_m: float | None = None,
) -> SearchResult:
    """Localize one frame by trying every grid centre × tile span and keeping the
    candidate with the most RANSAC inliers (ties: first encountered, so the order
    is deterministic). `fetch_tile` is injected so the same search runs against a
    live provider, a `TileCache`, or a synthetic world in tests.

    Two gates decide `locked`:
    - absolute: `best.n_inliers >= min_inliers_lock` (the sparse-matcher floor);
    - relative-margin: the winning peak must beat the strongest spatially-distinct
      rival by `lock_margin_ratio` (`best.n_inliers >= ratio * runner_up.n_inliers`).
      This is what makes DENSE matchers (RoMA) usable — they score high inliers on
      every tile, so only the MARGIN separates a true lock from a confident wrong
      one. `lock_margin_ratio=1.0` (default) makes this gate inert, preserving
      sparse-matcher behaviour. A rival counts only if its centre is at least
      `margin_separation_m` away (defaults to 1.5 x grid_step_m, so the immediate
      8-neighbour ring of the peak is not mistaken for a competitor)."""
    candidates: list[Candidate] = []
    best: Candidate | None = None
    for lat, lon in grid_centers(prior_lat, prior_lon, search_radius_m, grid_step_m):
        for span in scales_m:
            try:
                tile = fetch_tile(lat, lon, span, pixels)
                pose, _ = localize_frame(frame_bgr, tile, matcher)
            except ImportError:
                # Missing matcher dependency (e.g. torch/kornia for LoFTR) is a
                # setup error, not a bad tile — surface it loudly instead of
                # masking it as "0 inliers" on every cell.
                raise
            except Exception:
                # One bad tile (provider hiccup, malformed response) must not
                # kill the whole search — record the cell as failed and move on.
                candidates.append(Candidate(lat, lon, span))
                continue
            cand = Candidate(lat, lon, span, pose.n_inliers, pose.n_matches,
                             pose if pose.ok else None)
            candidates.append(cand)
            if cand.pose is not None and (best is None or cand.n_inliers > best.n_inliers):
                best = cand

    runner_up: Candidate | None = None
    margin_ratio: float | None = None
    locked = best is not None and best.n_inliers >= min_inliers_lock
    if best is not None:
        sep = margin_separation_m if margin_separation_m is not None else 1.5 * grid_step_m
        runner_up = _runner_up(best, candidates, sep)
        if runner_up is not None:
            # inf when the rival has zero inliers: an uncontested peak always passes.
            margin_ratio = (best.n_inliers / runner_up.n_inliers
                            if runner_up.n_inliers > 0 else float("inf"))
            if best.n_inliers < lock_margin_ratio * runner_up.n_inliers:
                locked = False
    return SearchResult(locked=locked, best=best, candidates=candidates,
                        runner_up=runner_up, margin_ratio=margin_ratio)


class TileCache:
    """Memoise a `FetchTile` in memory: the grid search re-requests the same
    centre × span tiles for every frame of the video, so even a plain dict cuts
    provider traffic by the number of frames. Lat/lon are rounded to 1e-7 deg
    (~1 cm) so float noise from the mercator roundtrip can't split keys."""

    def __init__(self, fetch_tile: FetchTile):
        self._fetch = fetch_tile
        self._tiles: dict[tuple[float, float, float, int], GeoImage] = {}

    def __call__(self, lat: float, lon: float, span_m: float, pixels: int) -> GeoImage:
        key = (round(lat, 7), round(lon, 7), span_m, pixels)
        tile = self._tiles.get(key)
        if tile is None:
            tile = self._fetch(lat, lon, span_m, pixels)
            self._tiles[key] = tile
        return tile
