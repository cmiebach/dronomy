"""Small working piece #7: multi-frame validation against the GPS ground truth.

Localizes a set of frames from the single coarse prior and scores the error
distribution against the telemetry track (ground truth ONLY — never an input).

Usage:
    python scripts/07_validate.py                          # 12 frames evenly spread
    python scripts/07_validate.py --frames 342,3083,6510   # explicit frames
    python scripts/07_validate.py --method loftr --provider pnoa
"""
import argparse

import _bootstrap  # noqa: F401
from dronomy_loc.config import load_config, resolve
from dronomy_loc.data import frames as frames_mod
from dronomy_loc.data.telemetry import load_track_csv
from dronomy_loc.localize.search import TileCache
from dronomy_loc.localize.validate import (
    grab_frames, make_world_fetch, parse_frames_spec, validate_frames,
    write_validation_csv,
)
from dronomy_loc.matching import get_matcher
from dronomy_loc.reference import get_provider


def _fmt_row(r) -> str:
    if r.locked:
        err = f"err {r.err_m:8.2f} m"
    elif r.err_m is not None:
        err = f"UNLOCKED (est err {r.err_m:.1f} m)"
    else:
        err = "UNLOCKED (no pose)"
    return (f"frame {r.frame:6d}  {err:32s}  inliers {r.n_inliers:4d}  "
            f"{r.runtime_s:6.2f} s")


def main():
    cfg = load_config()
    s = getattr(cfg.matching, "search", None)
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="12",
                    help="'342,3083,6510' = explicit | bare count '12' = evenly spread")
    ap.add_argument("--method", default=cfg.matching.method,
                    choices=["classical", "loftr"])
    ap.add_argument("--provider", default=cfg.reference.provider)
    ap.add_argument("--prior-lat", type=float, default=cfg.video.rough_lat)
    ap.add_argument("--prior-lon", type=float, default=cfg.video.rough_lon)
    ap.add_argument("--radius", type=float, default=getattr(s, "radius_m", 120.0))
    ap.add_argument("--step", type=float, default=getattr(s, "step_m", 60.0))
    ap.add_argument("--scales", default=",".join(
        str(x) for x in getattr(s, "scales_m", [50.0, 80.0, 110.0, 140.0])),
        help="comma-separated tile spans in meters")
    ap.add_argument("--pixels", type=int, default=getattr(s, "pixels", 640))
    ap.add_argument("--min-inliers", type=int, default=getattr(s, "min_inliers_lock", 20))
    ap.add_argument("--gps-track",
                    default=getattr(cfg.video, "gps_track_csv", "data/gps_track.csv"))
    ap.add_argument("--out", default="data/outputs/validation.csv")
    ap.add_argument("--world-span", type=float, default=600.0,
                    help="side of the ONE world tile (m); telemetry shows the whole "
                         "flight stays within ~109 m of the prior")
    ap.add_argument("--world-pixels", type=int, default=4096)
    args = ap.parse_args()

    track = load_track_csv(resolve(args.gps_track))
    video = resolve(cfg.video.path)
    meta = frames_mod.probe(video)
    n_total = meta["n_frames"] if meta["n_frames"] > 0 else max(f.frame for f in track) + 1
    fps = meta["fps"] or 29.97

    indices = parse_frames_spec(args.frames, n_total)
    print(f"Frames to validate ({len(indices)}): {indices}")
    print("Grabbing frames (one sequential pass)...")
    frames = grab_frames(video, indices,
                         resize_long_edge=getattr(cfg.frames, "resize_long_edge", 1920))

    # KEY OPTIMIZATION: the whole flight fits in one tile, so fetch the imagery
    # ONCE and serve every grid x scale candidate as a local crop of it.
    provider = get_provider(args.provider, cfg)
    print(f"Fetching one {args.world_span:g} m world tile "
          f"({args.world_pixels} px, provider={args.provider})...")
    world = provider.fetch(args.prior_lat, args.prior_lon,
                           args.world_span, args.world_pixels)
    fetch_tile = TileCache(make_world_fetch(world))   # shared across ALL frames

    matcher = get_matcher(args.method, cfg)
    scales = tuple(float(x) for x in args.scales.split(","))

    summary = validate_frames(
        frames, track, args.prior_lat, args.prior_lon, matcher, fetch_tile,
        fps=fps, search_radius_m=args.radius, grid_step_m=args.step,
        scales_m=scales, pixels=args.pixels, min_inliers_lock=args.min_inliers,
        on_row=lambda r: print(_fmt_row(r)),
    )

    print("---- validation summary ----")
    print(f"frames     : {summary.n}")
    print(f"locked     : {summary.n_locked} ({100.0 * summary.lock_rate:.1f}%)")
    if summary.n_locked:
        print(f"median err : {summary.median_err_m:.2f} m")
        print(f"mean err   : {summary.mean_err_m:.2f} m")
        print(f"worst err  : {summary.worst_err_m:.2f} m")
        print("(error stats over LOCKED frames only; "
              f"{summary.n - summary.n_locked} unlocked excluded)")
    else:
        print("WARNING: no frame locked -- no error statistics. The CSV still "
              "records every attempt; reporting failure honestly is the job.")

    out = write_validation_csv(summary, resolve(args.out))
    print(f"Wrote {len(summary.rows)} rows -> {out}")


if __name__ == "__main__":
    main()
