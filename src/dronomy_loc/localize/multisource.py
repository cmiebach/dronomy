"""Per-frame imagery-source selection.

Different satellite basemaps lock different frames: measured on this flight,
PNOA and Esri are complementary — Esri sharpened some frames (19 m -> 5 m) and
gave more inliers, while PNOA locked frames Esri missed. So instead of committing
to one provider, localize each frame against SEVERAL and keep the most confident
lock. This is the imagery analogue of the matcher auto-selection: it lifts BOTH
coverage (union of what any source can register) and accuracy (the most
confident fix wins), with no per-scene tuning.

Telemetry-free: GPS is used only to score the chosen fix, never to choose it —
selection is by lock confidence (inlier count), an input-only signal.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable

from ..data.telemetry import gt_for_frame
from ..reference.geo import haversine_m
from .search import FetchTile, search_localize
from .validate import FrameScore, ValidationSummary


@dataclass
class ProviderChoice:
    """Which provider won a frame, and how confidently."""
    frame: int
    provider: str | None        # winning provider name, or None if nothing locked
    n_inliers: int


def localize_multisource(frame_bgr, prior_lat, prior_lon, matcher,
                         providers: dict[str, FetchTile], *,
                         search_radius_m: float = 120.0, grid_step_m: float = 60.0,
                         scales_m: tuple[float, ...] = (50.0, 80.0, 110.0, 140.0),
                         pixels: int = 640, min_inliers_lock: int = 20):
    """Localize one frame against every provider; return (best_result, name).

    The winner is the LOCKED candidate with the most inliers across providers
    (confidence-based, telemetry-free). Returns (None, None) if none lock."""
    best_res, best_name = None, None
    for name, fetch in providers.items():
        res = search_localize(frame_bgr, prior_lat, prior_lon, matcher, fetch,
                              search_radius_m=search_radius_m, grid_step_m=grid_step_m,
                              scales_m=scales_m, pixels=pixels,
                              min_inliers_lock=min_inliers_lock)
        if res.locked and res.best is not None:
            if best_res is None or res.best.n_inliers > best_res.best.n_inliers:
                best_res, best_name = res, name
    return best_res, best_name


def validate_multisource(frames_by_idx, track, prior_lat, prior_lon, matcher,
                         providers: dict[str, FetchTile], *, fps: float = 29.97,
                         search_radius_m: float = 120.0, grid_step_m: float = 60.0,
                         scales_m: tuple[float, ...] = (50.0, 80.0, 110.0, 140.0),
                         pixels: int = 640, min_inliers_lock: int = 20,
                         on_row: Callable[[FrameScore, str | None], None] | None = None
                         ) -> tuple[ValidationSummary, list[ProviderChoice]]:
    """Multi-source version of `validate_frames`: each frame is localized against
    all `providers` and scored on its best lock. Returns the usual summary plus
    the per-frame provider choice (so you can report which source won where)."""
    rows: list[FrameScore] = []
    choices: list[ProviderChoice] = []
    for idx in sorted(frames_by_idx):
        gt = gt_for_frame(track, idx)
        t0 = time.perf_counter()
        best, name = localize_multisource(
            frames_by_idx[idx], prior_lat, prior_lon, matcher, providers,
            search_radius_m=search_radius_m, grid_step_m=grid_step_m,
            scales_m=scales_m, pixels=pixels, min_inliers_lock=min_inliers_lock)
        runtime = time.perf_counter() - t0
        pose = best.best.pose if best is not None else None
        row = FrameScore(
            frame=idx, t_s=idx / fps,
            est_lat=pose.lat if pose else None, est_lon=pose.lon if pose else None,
            gt_lat=gt.lat, gt_lon=gt.lon,
            err_m=haversine_m(gt.lat, gt.lon, pose.lat, pose.lon) if pose else None,
            yaw_deg=pose.yaw_deg if pose else None,
            n_inliers=best.best.n_inliers if best is not None else 0,
            locked=best is not None, runtime_s=runtime)
        rows.append(row)
        choices.append(ProviderChoice(idx, name, row.n_inliers))
        if on_row is not None:
            on_row(row, name)

    locked = [r.err_m for r in rows if r.locked and r.err_m is not None]
    n = len(rows)
    summary = ValidationSummary(
        n=n, n_locked=len(locked),
        lock_rate=len(locked) / n if n else 0.0,
        median_err_m=statistics.median(locked) if locked else None,
        mean_err_m=statistics.fmean(locked) if locked else None,
        worst_err_m=max(locked) if locked else None, rows=rows)
    return summary, choices
