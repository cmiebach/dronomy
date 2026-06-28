"""Small working piece #8: full-trajectory estimate via VO dead-reckoning.

Anchors (absolute fixes from grid-search matching, telemetry-free) reset the
drift of a frame-to-frame homography chain, extending localization to frames
that cannot match the satellite map directly. Output: per-frame estimates +
error-vs-hops-from-anchor curve scored against the GPS ground-truth track
(GT is scoring only, never an input).

Usage:
    python scripts/08_vo_trajectory.py --provider pnoa --anchors 4040,6500
    python scripts/08_vo_trajectory.py --stride 30 --method loftr
"""
import argparse
import csv
import math
import time
from types import SimpleNamespace

import _bootstrap  # noqa: F401
import cv2

from dronomy_loc.config import load_config, resolve
from dronomy_loc.data.frames import _resize_long_edge
from dronomy_loc.data.telemetry import gt_for_frame, load_track_csv
from dronomy_loc.localize import (
    TileCache, anchor_from, chain_poses, drift_curve, grab_frames, localize_frame,
    make_world_fetch, pairwise_homographies, search_localize,
)
from dronomy_loc.matching import get_matcher
from dronomy_loc.reference import get_provider, load_reference, save_reference
from dronomy_loc.reference.geo import haversine_m


def main():
    cfg = load_config()
    s = getattr(cfg.matching, "search", SimpleNamespace())
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=cfg.video.path)
    ap.add_argument("--provider", default=cfg.reference.provider,
                    choices=["esri", "pnoa", "gee", "ign"])
    ap.add_argument("--method", default="loftr", choices=["classical", "loftr"],
                    help="matcher for the ANCHOR search (VO links always use SIFT)")
    ap.add_argument("--anchors", default="4040,6500",
                    help="candidate anchor keyframes (must lie on the stride grid)")
    ap.add_argument("--stride", type=int, default=10,
                    help="sweep every Nth frame (10 = 3 fps, ~0.3 m of motion)")
    ap.add_argument("--resize", type=int, default=960,
                    help="long edge for sweep AND anchor frames (must match)")
    ap.add_argument("--radius", type=float, default=getattr(s, "radius_m", 120.0))
    ap.add_argument("--step", type=float, default=40.0)
    ap.add_argument("--scales", default="70",
                    help="anchor-search tile spans, m (70 = measured footprint)")
    ap.add_argument("--pixels", type=int, default=getattr(s, "pixels", 640))
    ap.add_argument("--min-inliers", type=int, default=getattr(s, "min_inliers_lock", 20))
    ap.add_argument("--vo-min-inliers", type=int, default=30,
                    help="pairwise link acceptance (consecutive frames overlap hugely)")
    ap.add_argument("--gps-track", default=getattr(cfg.video, "gps_track_csv",
                                                   "data/gps_track.csv"))
    ap.add_argument("--out", default=getattr(cfg.output, "vo_trajectory_csv",
                                             "data/outputs/vo_trajectory.csv"))
    args = ap.parse_args()

    video = resolve(args.video)
    track = load_track_csv(resolve(args.gps_track))
    prior_lat, prior_lon = cfg.video.rough_lat, cfg.video.rough_lon
    scales = tuple(float(v) for v in args.scales.split(","))
    anchor_idxs = sorted(int(v) for v in args.anchors.split(","))
    bad = [a for a in anchor_idxs if a % args.stride]
    if bad:
        raise SystemExit(f"anchor frames {bad} not on the stride-{args.stride} grid")

    # One world tile; every candidate is a local crop (zero network per cell).
    # Cached on disk so a flaky WMS (PNOA 502s observed live) can't stall runs.
    span = getattr(cfg.reference, "world_span_m", 600.0)
    px = getattr(cfg.reference, "world_pixels", 4096)
    ref_dir = resolve(cfg.reference.out_dir)
    cache_name = f"world_{args.provider}"
    try:
        world = load_reference(ref_dir, cache_name)
        print(f"Loaded cached world tile reference_{cache_name}.png")
    except FileNotFoundError:
        provider = get_provider(args.provider, cfg)
        print(f"Fetching {args.provider} world tile ({span:g} m @ {px}px) ...")
        world = provider.fetch(prior_lat, prior_lon, span, px)
        save_reference(world, ref_dir, cache_name)
    fetch = TileCache(make_world_fetch(world))

    # -- Anchor pass: telemetry-free grid search on the candidate keyframes --
    anchor_matcher = get_matcher(args.method, cfg)
    anchor_frames = grab_frames(video, anchor_idxs, resize_long_edge=args.resize)
    anchors = []
    for idx in anchor_idxs:
        t0 = time.time()
        res = search_localize(anchor_frames[idx], prior_lat, prior_lon,
                              anchor_matcher, fetch,
                              search_radius_m=args.radius, grid_step_m=args.step,
                              scales_m=scales, pixels=args.pixels,
                              min_inliers_lock=args.min_inliers)
        dt = time.time() - t0
        if not res.locked:
            print(f"anchor {idx}: NOT LOCKED "
                  f"(best {res.best.n_inliers if res.best else 0} inliers, {dt:.0f}s)")
            continue
        tile = fetch(res.best.lat, res.best.lon, res.best.span_m, args.pixels)
        pose, mr = localize_frame(anchor_frames[idx], tile, anchor_matcher)
        if not mr.ok or mr.n_inliers < args.min_inliers:
            print(f"anchor {idx}: re-match below gate, skipped")
            continue
        anchors.append(anchor_from(idx, mr.homography, tile))
        gt = gt_for_frame(track, idx)
        err = haversine_m(pose.lat, pose.lon, gt.lat, gt.lon)
        print(f"anchor {idx}: LOCKED err={err:.2f} m inliers={mr.n_inliers} ({dt:.0f}s)")
    if not anchors:
        raise SystemExit("No anchor locked -- nothing to chain from.")

    # -- Sweep pass: stream every Nth frame, build consecutive VO links --
    vo_cfg = SimpleNamespace(matching=SimpleNamespace(
        classical=SimpleNamespace(detector="SIFT", max_features=3000, ratio_test=0.75),
        ransac=SimpleNamespace(reproj_threshold_px=5.0, confidence=0.999, min_inliers=12)))
    vo_matcher = get_matcher("classical", vo_cfg)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    links, prev, shape = [], None, None
    idx = n_swept = 0
    t0 = time.time()
    while True:
        if not cap.grab():
            break
        if idx % args.stride == 0:
            ok, img = cap.retrieve()
            if not ok:
                break
            img = _resize_long_edge(img, args.resize)
            shape = img.shape
            n_swept += 1
            if prev is not None:
                links.append(pairwise_homographies(
                    [prev, (idx, img)], vo_matcher,
                    min_inliers=args.vo_min_inliers)[0])
            prev = (idx, img)
        idx += 1
    cap.release()
    n_breaks = sum(1 for l in links if l.H is None)
    print(f"sweep: {n_swept} frames, {len(links)} links "
          f"({n_breaks} breaks) in {time.time()-t0:.0f}s")

    # -- Chain + score --
    chain = chain_poses(links, anchors, shape[:2])
    rows = drift_curve(chain, track)
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["frame", "hops_from_anchor", "anchor_frame", "err_m",
            "est_lat", "est_lon", "gt_lat", "gt_lon"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    errs = [r["err_m"] for r in rows]
    errs_sorted = sorted(errs)
    rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
    print(f"coverage: {len(rows)}/{n_swept} swept frames "
          f"({100.0 * len(rows) / n_swept:.0f}%)")
    print(f"error vs GT: median={errs_sorted[len(errs)//2]:.1f} m  "
          f"mean={sum(errs)/len(errs):.1f} m  rmse={rmse:.1f} m  "
          f"worst={errs_sorted[-1]:.1f} m")
    for lo, hi in ((0, 10), (11, 50), (51, 10**9)):
        bucket = [r["err_m"] for r in rows if lo <= r["hops_from_anchor"] <= hi]
        if bucket:
            b = sorted(bucket)
            label = f"{lo}-{hi if hi < 10**9 else 'max'}"
            print(f"  hops {label:>7}: n={len(b):4d}  median={b[len(b)//2]:7.1f} m  "
                  f"worst={b[-1]:7.1f} m")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
