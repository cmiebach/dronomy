"""Multi-frame validation harness: localize frames from one coarse prior and
score the error distribution against the GPS ground-truth track.

This is the instrument every accuracy claim in the report is measured with, so
determinism and honest reporting beat features: frames are processed in sorted
order, errors are summarised over LOCKED frames only (an unlocked estimate is
not an accuracy claim — it is reported, but separately), and "no lock at all"
yields None statistics instead of a flattering empty mean.

The telemetry showed the whole flight stays within ~109 m of the prior, so the
entire grid x scale search fits inside ONE world tile; `make_world_fetch` turns
that tile into a `FetchTile` that crops locally — zero per-candidate network
calls and bit-identical tiles run to run.
"""
from __future__ import annotations

import csv
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..data.frames import _resize_long_edge
from ..data.telemetry import GPSFix, gt_for_frame
from ..matching.base import Matcher
from ..reference.geo import GeoImage, haversine_m, mercator_bbox_around
from .search import FetchTile, search_localize


@dataclass
class FrameScore:
    """One frame's verdict. `err_m` is None when no pose was estimated at all;
    an UNLOCKED frame with a pose still gets its err_m recorded (honesty), but
    only LOCKED frames enter the summary statistics."""
    frame: int
    t_s: float
    est_lat: float | None
    est_lon: float | None
    gt_lat: float
    gt_lon: float
    err_m: float | None
    yaw_deg: float | None
    n_inliers: int
    locked: bool
    runtime_s: float


@dataclass
class ValidationSummary:
    """Error stats are over LOCKED frames only; all None when nothing locked."""
    n: int
    n_locked: int
    lock_rate: float
    median_err_m: float | None
    mean_err_m: float | None
    worst_err_m: float | None
    rows: list[FrameScore]


def parse_frames_spec(spec: str, n_total: int) -> list[int]:
    """'342,3083,6510' -> those exact frames; a bare count '12' -> 12 indices
    evenly spread across [0, n_total-1]. Returns sorted unique indices either
    way (one sequential grab pass needs them ordered). Junk raises ValueError."""
    s = spec.strip()
    if n_total <= 0:
        raise ValueError(f"n_total must be positive, got {n_total}")
    try:
        if "," in s:
            idxs = sorted({int(tok) for tok in s.split(",") if tok.strip()})
        else:
            count = int(s)
            if count < 1:
                raise ValueError(f"frame count must be >= 1, got {count}")
            if count == 1:
                idxs = [0]
            else:
                idxs = sorted({round(k * (n_total - 1) / (count - 1))
                               for k in range(count)})
    except ValueError as e:
        raise ValueError(f"bad frames spec {spec!r}: {e}") from None
    if not idxs:
        raise ValueError(f"bad frames spec {spec!r}: no frame indices")
    if idxs[0] < 0 or idxs[-1] >= n_total:
        raise ValueError(f"frame indices {idxs} out of range [0, {n_total - 1}]")
    return idxs


def grab_frames(
    video_path: str | Path,
    indices: list[int],
    resize_long_edge: int | None = 1920,
) -> dict[int, np.ndarray]:
    """Decode the requested frames in ONE sequential grab()/retrieve() pass.
    No CAP_PROP_POS_* seeking — flaky on Windows (same convention as
    data/frames.py). Raises if any requested index can't be read."""
    wanted = sorted({int(i) for i in indices})
    if not wanted:
        return {}
    if wanted[0] < 0:
        raise ValueError(f"negative frame index in {wanted}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    out: dict[int, np.ndarray] = {}
    try:
        remaining = iter(wanted)
        target = next(remaining)
        idx = 0
        while True:
            if not cap.grab():             # cheap: advance without decoding
                break
            if idx == target:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                out[idx] = _resize_long_edge(frame, resize_long_edge)
                try:
                    target = next(remaining)
                except StopIteration:
                    break
            idx += 1
    finally:
        cap.release()
    missing = [i for i in wanted if i not in out]
    if missing:
        raise RuntimeError(f"could not grab frames {missing} from {video_path}")
    return out


def make_world_fetch(world: GeoImage) -> FetchTile:
    """Wrap one big georeferenced world tile as a `FetchTile` that CROPS locally.
    The bbox of each crop is re-derived from the actual (clamped, rounded) pixel
    rect, so edge tiles stay exactly georeferenced even when the requested
    square doesn't fit (their meter footprint just isn't square anymore).
    Raises when the request lies (almost) entirely outside the world."""
    def fetch(lat: float, lon: float, span_m: float, pixels: int) -> GeoImage:
        minx, miny, maxx, maxy = mercator_bbox_around(lon, lat, span_m)
        x0, y0 = world.mercator_to_pixel(minx, maxy)   # top-left (row 0 == maxy)
        x1, y1 = world.mercator_to_pixel(maxx, miny)
        x0, y0 = max(0, round(x0)), max(0, round(y0))
        x1, y1 = min(world.width, round(x1)), min(world.height, round(y1))
        if x1 - x0 < 2 or y1 - y0 < 2:
            raise ValueError(
                f"tile ({lat:.6f},{lon:.6f}) span={span_m:g}m outside the world tile")
        tile = cv2.resize(world.image[y0:y1, x0:x1], (pixels, pixels),
                          interpolation=cv2.INTER_AREA)
        gx0, gy0 = world.pixel_to_mercator(x0, y0)
        gx1, gy1 = world.pixel_to_mercator(x1, y1)
        return GeoImage(image=tile, bbox=(gx0, gy1, gx1, gy0))
    return fetch


def validate_frames(
    frames_by_idx: dict[int, np.ndarray],
    track: list[GPSFix],
    prior_lat: float,
    prior_lon: float,
    matcher: Matcher,
    fetch_tile: FetchTile,
    *,
    fps: float = 29.97,
    search_radius_m: float = 120.0,
    grid_step_m: float = 60.0,
    scales_m: tuple[float, ...] = (50.0, 80.0, 110.0, 140.0),
    pixels: int = 640,
    min_inliers_lock: int = 20,
    lock_margin_ratio: float = 1.0,
    margin_separation_m: float | None = None,
    on_row: Callable[[FrameScore], None] | None = None,
) -> ValidationSummary:
    """Run `search_localize` on every frame (sorted index order, deterministic)
    and score each estimate against `gt_for_frame`.

    `fetch_tile` must already be shared/cached by the caller (a `TileCache`
    around a provider, or `make_world_fetch(world)`): every frame re-requests
    the same grid x scale tiles, so wrapping it here per-call would defeat the
    reuse. `on_row` (optional) is called after each frame — for live printing."""
    rows: list[FrameScore] = []
    for idx in sorted(frames_by_idx):
        gt = gt_for_frame(track, idx)
        t0 = time.perf_counter()
        res = search_localize(
            frames_by_idx[idx], prior_lat, prior_lon, matcher, fetch_tile,
            search_radius_m=search_radius_m, grid_step_m=grid_step_m,
            scales_m=scales_m, pixels=pixels, min_inliers_lock=min_inliers_lock,
            lock_margin_ratio=lock_margin_ratio,
            margin_separation_m=margin_separation_m,
        )
        runtime = time.perf_counter() - t0
        pose = res.best.pose if res.best is not None else None
        row = FrameScore(
            frame=idx,
            t_s=idx / fps,
            est_lat=pose.lat if pose else None,
            est_lon=pose.lon if pose else None,
            gt_lat=gt.lat,
            gt_lon=gt.lon,
            err_m=haversine_m(gt.lat, gt.lon, pose.lat, pose.lon) if pose else None,
            yaw_deg=pose.yaw_deg if pose else None,
            n_inliers=res.best.n_inliers if res.best is not None else 0,
            locked=res.locked,
            runtime_s=runtime,
        )
        rows.append(row)
        if on_row is not None:
            on_row(row)

    locked_errs = [r.err_m for r in rows if r.locked and r.err_m is not None]
    n = len(rows)
    return ValidationSummary(
        n=n,
        n_locked=len(locked_errs),
        lock_rate=len(locked_errs) / n if n else 0.0,
        median_err_m=statistics.median(locked_errs) if locked_errs else None,
        mean_err_m=statistics.fmean(locked_errs) if locked_errs else None,
        worst_err_m=max(locked_errs) if locked_errs else None,
        rows=rows,
    )


_CSV_FIELDS = ["frame", "t_s", "est_lat", "est_lon", "gt_lat", "gt_lon",
               "err_m", "yaw_deg", "n_inliers", "locked", "runtime_s"]


def write_validation_csv(summary: ValidationSummary, path: str | Path) -> Path:
    """One row per FrameScore. Atomic (tmp + os.replace) so a crash can't leave
    a half-written results file; floats via str() round-trip exactly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(_CSV_FIELDS)
        for r in summary.rows:
            w.writerow([
                r.frame, r.t_s,
                "" if r.est_lat is None else r.est_lat,
                "" if r.est_lon is None else r.est_lon,
                r.gt_lat, r.gt_lon,
                "" if r.err_m is None else r.err_m,
                "" if r.yaw_deg is None else r.yaw_deg,
                r.n_inliers, int(r.locked), r.runtime_s,
            ])
    os.replace(tmp, path)
    return path


def read_validation_csv(path: str | Path) -> list[FrameScore]:
    """Inverse of `write_validation_csv` (exact round-trip, used by tests and
    any downstream plotting)."""
    rows: list[FrameScore] = []
    with open(path, "r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(FrameScore(
                frame=int(row["frame"]),
                t_s=float(row["t_s"]),
                est_lat=float(row["est_lat"]) if row["est_lat"] else None,
                est_lon=float(row["est_lon"]) if row["est_lon"] else None,
                gt_lat=float(row["gt_lat"]),
                gt_lon=float(row["gt_lon"]),
                err_m=float(row["err_m"]) if row["err_m"] else None,
                yaw_deg=float(row["yaw_deg"]) if row["yaw_deg"] else None,
                n_inliers=int(row["n_inliers"]),
                locked=bool(int(row["locked"])),
                runtime_s=float(row["runtime_s"]),
            ))
    return rows
