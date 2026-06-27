"""Cascade RoMA pass (runs INSIDE the matchanything Docker container).

RoMA is the expensive, high-coverage matcher. Running a full grid search with it
is infeasible under amd64 emulation, so we use the cascade the project is about:
the cheap matcher (LoFTR, already run on the host) locates each frame; RoMA then
runs at THAT location (1 candidate = refine) for frames LoFTR locked, and only
falls back to a SMALL grid to RECOVER the frames LoFTR missed. This bounds RoMA
to a handful of matches per frame instead of ~100.

Reads the host LoFTR CSV for per-frame centres, writes a validation CSV with the
same columns so the combine step can compare all methods uniformly.

Run (from repo root, repo mounted at CWD, PYTHONPATH=src):
  python scripts/bench_roma_cascade.py --loftr-csv data/outputs/val_loftr_12.csv \
      --out data/outputs/val_roma_12.csv
"""
import argparse, csv, sys, time
from pathlib import Path

from dronomy_loc.data.telemetry import load_track_csv, gt_for_frame
from dronomy_loc.localize.validate import grab_frames, make_world_fetch, parse_frames_spec
from dronomy_loc.localize.search import search_localize, TileCache
from dronomy_loc.reference.store import load_reference
from dronomy_loc.reference.geo import haversine_m
from dronomy_loc.matching.matchanything import MatchAnythingMatcher

FIELDS = ["frame", "t_s", "est_lat", "est_lon", "gt_lat", "gt_lon",
          "err_m", "yaw_deg", "n_inliers", "locked", "runtime_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="dronomy_video/IE_Challenge_lat43_521955_lon5_624290.MP4")
    ap.add_argument("--frames-dir", default=None,
                    help="dir of pre-extracted frames named frame_<idx>.jpg (skips video decode)")
    ap.add_argument("--gps-track", default="data/gps_track.csv")
    ap.add_argument("--ref-dir", default="data/reference")
    ap.add_argument("--ref-name", default="world_pnoa")
    ap.add_argument("--loftr-csv", default="data/outputs/val_loftr_12.csv")
    ap.add_argument("--spread", type=int, default=12)
    ap.add_argument("--prior-lat", type=float, default=43.521955)
    ap.add_argument("--prior-lon", type=float, default=-5.624290)
    ap.add_argument("--pixels", type=int, default=640)
    ap.add_argument("--min-inliers", type=int, default=20)
    ap.add_argument("--recover-radius", type=float, default=60.0)
    ap.add_argument("--recover-step", type=float, default=60.0)
    ap.add_argument("--max-recover", type=int, default=3,
                    help="cap on RoMA grid-search (recover) frames — bounds emulated cost")
    ap.add_argument("--scale", type=float, default=110.0)
    ap.add_argument("--fps", type=float, default=29.97)
    ap.add_argument("--device", default="cpu", help="cpu | cuda | mps (RoMA backend)")
    ap.add_argument("--out", default="data/outputs/val_roma_12.csv")
    args = ap.parse_args()

    track = load_track_csv(args.gps_track)
    n_total = max(f.frame for f in track) + 1
    indices = parse_frames_spec(str(args.spread), n_total)
    print(f"RoMA cascade on {len(indices)} frames: {indices}", flush=True)

    # Per-frame centres from the host LoFTR pass (cascade hand-off).
    centres = {}
    if Path(args.loftr_csv).exists():
        for r in csv.DictReader(open(args.loftr_csv)):
            if r.get("locked") in ("1", "True", "true") and r.get("est_lat"):
                centres[int(r["frame"])] = (float(r["est_lat"]), float(r["est_lon"]))
        print(f"LoFTR provided {len(centres)} locked centres (refine); "
              f"{len(indices) - len(centres)} frames need RoMA recovery search", flush=True)

    if args.frames_dir:                          # pre-extracted jpgs (no video on host)
        import cv2
        frames = {}
        for idx in indices:
            p = Path(args.frames_dir) / f"frame_{idx}.jpg"
            img = cv2.imread(str(p))
            if img is None:
                print(f"frame {idx}: missing {p}", flush=True); continue
            frames[idx] = img
    else:
        frames = grab_frames(args.video, indices)
    world = load_reference(args.ref_dir, args.ref_name)
    fetch_tile = TileCache(make_world_fetch(world))
    matcher = MatchAnythingMatcher(None, model="roma")
    matcher.device = args.device                 # cpu (local) | cuda (RunPod GPU)
    print(f"RoMA device: {matcher.device}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    recover_used = 0
    for idx in sorted(frames):
        gt = gt_for_frame(track, idx)
        if idx in centres:                       # REFINE: 1 candidate at LoFTR's lock
            clat, clon = centres[idx]
            radius, step = 0.0, 1000.0
            mode = "refine"
        elif recover_used < args.max_recover:    # RECOVER: small grid at the prior (capped)
            clat, clon = args.prior_lat, args.prior_lon
            radius, step = args.recover_radius, args.recover_step
            mode = "recover"
            recover_used += 1
        else:                                    # over the recover budget — skip (honest)
            row = dict(frame=idx, t_s=idx / args.fps, est_lat="", est_lon="",
                       gt_lat=gt.lat, gt_lon=gt.lon, err_m="", yaw_deg="",
                       n_inliers=0, locked=0, runtime_s=0.0)
            rows.append(row)
            print(f"frame {idx:5d} [skip   ] recover budget spent ({args.max_recover})", flush=True)
            with open(out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
            continue
        t0 = time.perf_counter()
        try:
            res = search_localize(frames[idx], clat, clon, matcher, fetch_tile,
                                  search_radius_m=radius, grid_step_m=step,
                                  scales_m=(args.scale,), pixels=args.pixels,
                                  min_inliers_lock=args.min_inliers)
            pose = res.best.pose if res.best is not None else None
            err = haversine_m(gt.lat, gt.lon, pose.lat, pose.lon) if pose else None
            row = dict(frame=idx, t_s=idx / args.fps,
                       est_lat=pose.lat if pose else "", est_lon=pose.lon if pose else "",
                       gt_lat=gt.lat, gt_lon=gt.lon,
                       err_m=err if err is not None else "",
                       yaw_deg=pose.yaw_deg if pose else "",
                       n_inliers=res.best.n_inliers if res.best else 0,
                       locked=int(res.locked), runtime_s=time.perf_counter() - t0)
        except Exception as e:                   # never lose the whole run to one frame
            row = dict(frame=idx, t_s=idx / args.fps, est_lat="", est_lon="",
                       gt_lat=gt.lat, gt_lon=gt.lon, err_m="", yaw_deg="",
                       n_inliers=0, locked=0, runtime_s=time.perf_counter() - t0)
            print(f"frame {idx}: ERROR {e}", flush=True)
        rows.append(row)
        e = row["err_m"]
        print(f"frame {idx:5d} [{mode:7s}] locked={row['locked']} "
              f"inliers={row['n_inliers']:4d} err={e if e!='' else 'None'} "
              f"{row['runtime_s']:.0f}s", flush=True)
        with open(out, "w", newline="") as fh:   # rewrite each frame = crash-safe
            w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)

    locked = [r for r in rows if r["locked"]]
    print(f"\nRoMA: {len(locked)}/{len(rows)} locked. Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
