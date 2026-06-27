"""Localization models — a uniform wrapper over (matcher + grid search).

A `LocalizationModel` turns the existing engine (`get_matcher` + `search_localize`)
into one call: `localize(sample, fetch_tile, prior) -> FrameScore`. This is the
interchangeable unit the runner benchmarks; the registry (`get_model`) names them
`sift | loftr | roma | eloftr`. We WRAP the engine, never reimplement it.

GROUND-TRUTH RULE: localization reads ONLY `sample.image_bgr` and the coarse
`prior` (the rough known launch area — not telemetry). `sample.gt` is touched
solely to fill the score row's truth/error fields, never to drive the search.

Per-scene search scaling (the critical real-data finding): the grid radius and
tile spans must match the flight's altitude/footprint. The provided flight is
~50 m AGL (~71 m footprint); UAV-VisLoc region 03 is ~466 m (~840 m footprint),
and the 50 m defaults will not lock there. `search_for_altitude` derives a
`SceneSearch` from altitude + field of view so one runner serves both.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from ..localize.search import FetchTile, search_localize
from ..localize.validate import FrameScore
from ..matching.base import Matcher, get_matcher
from ..reference.geo import haversine_m

MODEL_NAMES = ("sift", "loftr", "roma", "eloftr")


@dataclass
class SceneSearch:
    """Grid-search parameters for one scenario (altitude/footprint dependent)."""
    radius_m: float = 120.0
    grid_step_m: float = 60.0
    scales_m: tuple[float, ...] = (50.0, 80.0, 110.0, 140.0)
    pixels: int = 640
    min_inliers_lock: int = 20


def _hfov_deg(hfov_deg: float | None, focal_px: float | None,
              image_width_px: float | None) -> float | None:
    """Horizontal FOV from an explicit value, or derived from focal length +
    image width: hfov = 2*atan(W / (2*f))."""
    if hfov_deg is not None:
        return hfov_deg
    if focal_px and image_width_px:
        return math.degrees(2.0 * math.atan(image_width_px / (2.0 * focal_px)))
    return None


def search_for_altitude(
    altitude_m: float,
    *,
    hfov_deg: float | None = None,
    focal_px: float | None = None,
    image_width_px: float | None = None,
    min_inliers_lock: int = 20,
    pixels: int = 640,
) -> SceneSearch:
    """Derive scene-appropriate search parameters from the expected altitude.

    Footprint ~= 2 * altitude * tan(hfov/2). We centre the tile spans on the
    footprint (the true scale is unknown, so we bracket it 0.6x-1.4x), set the
    search radius to ~one footprint (covers a prior that is off by up to a frame
    width), and the grid step to ~half a footprint. Falls back to the 50 m-flight
    defaults when neither FOV nor focal length is known.
    """
    fov = _hfov_deg(hfov_deg, focal_px, image_width_px)
    if not altitude_m or altitude_m <= 0 or fov is None:
        return SceneSearch(min_inliers_lock=min_inliers_lock, pixels=pixels)
    footprint = 2.0 * altitude_m * math.tan(math.radians(fov) / 2.0)
    scales = tuple(round(footprint * f, 1) for f in (0.6, 0.85, 1.1, 1.4))
    return SceneSearch(
        radius_m=round(footprint, 1),
        grid_step_m=round(footprint / 2.0, 1),
        scales_m=scales,
        pixels=pixels,
        min_inliers_lock=min_inliers_lock,
    )


@dataclass
class LocalizationModel:
    """A named matcher + its scene search params, with a uniform localize()."""
    name: str
    matcher: Matcher
    search: SceneSearch = field(default_factory=SceneSearch)

    def localize(self, sample, fetch_tile: FetchTile,
                 prior: tuple[float, float],
                 search: "SceneSearch | None" = None) -> FrameScore:
        """Localize one Sample against a reference `fetch_tile` from a coarse
        `prior` (lat, lon). `search` overrides the model's default per call (the
        runner passes a scene-scaled SceneSearch). Returns a FrameScore;
        `sample.gt` is used only to populate the truth/error fields, never to
        drive the search."""
        s = search or self.search
        t0 = time.perf_counter()
        res = search_localize(
            sample.image_bgr, prior[0], prior[1], self.matcher, fetch_tile,
            search_radius_m=s.radius_m, grid_step_m=s.grid_step_m,
            scales_m=s.scales_m, pixels=s.pixels,
            min_inliers_lock=s.min_inliers_lock,
        )
        runtime = time.perf_counter() - t0
        pose = res.best.pose if res.best is not None else None
        gt = sample.gt
        gt_lat = gt.lat if gt is not None else math.nan
        gt_lon = gt.lon if gt is not None else math.nan
        err = (haversine_m(gt.lat, gt.lon, pose.lat, pose.lon)
               if (pose is not None and gt is not None) else None)
        return FrameScore(
            frame=sample.frame_id,
            t_s=sample.t_s if sample.t_s is not None else float(sample.frame_id),
            est_lat=pose.lat if pose else None,
            est_lon=pose.lon if pose else None,
            gt_lat=gt_lat,
            gt_lon=gt_lon,
            err_m=err,
            yaw_deg=pose.yaw_deg if pose else None,
            n_inliers=res.best.n_inliers if res.best is not None else 0,
            locked=res.locked,
            runtime_s=runtime,
        )


def get_model(name: str, cfg=None, search: SceneSearch | None = None) -> LocalizationModel:
    """Registry: 'sift' | 'loftr' | 'roma' | 'eloftr'.

    `roma`/`eloftr` are MatchAnything variants (real weights live in the Docker
    image; the import is deferred so this works offline). `search` overrides the
    default SceneSearch — pass `search_for_altitude(...)` per scenario.
    """
    key = name.lower()
    if key in ("sift", "classical"):
        matcher = get_matcher("classical", cfg)
    elif key == "loftr":
        matcher = get_matcher("loftr", cfg)
    elif key in ("roma", "eloftr"):
        from ..matching.matchanything import MatchAnythingMatcher
        matcher = MatchAnythingMatcher(cfg, model=key)
    else:
        raise ValueError(f"Unknown model: {name!r} (known: {MODEL_NAMES})")
    return LocalizationModel(name=key, matcher=matcher,
                             search=search or SceneSearch())
