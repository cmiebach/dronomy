"""FULL pipeline over the whole drone video -> one continuous flight path.

This is the end-to-end run the project is about:
  1. Sweep EVERY Nth frame of the whole video; SIFT visual odometry chains
     consecutive frames into a continuous relative track.
  2. At anchor keyframes, run MULTIPLE matchers (SIFT + LoFTR) against the
     satellite map and keep the most-confident absolute lock — PER-FRAME matcher
     selection (the framework's point).
  3. Chain the VO onto the absolute anchors -> one continuous, georeferenced
     flight path over the entire video, scored against GPS.

Per-frame matcher = whichever locks with the most inliers (telemetry-free; GPS
only scores). RoMA is intentionally NOT in this whole-video loop: it is GPU-only
and ~minutes/match, so it cannot run per-frame over thousands of frames on
available hardware (documented limitation, not a design choice).

  python scripts/full_pipeline.py --stride 10 --anchors 28 --device mps
"""
from __future__ import annotations

import argparse, csv, time
from pathlib import Path
from types import SimpleNamespace

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
from dronomy_loc.config import load_config
from dronomy_loc.data.frames import _resize_long_edge
from dronomy_loc.data.telemetry import load_track_csv, gt_for_frame
from dronomy_loc.localize.validate import grab_frames, make_world_fetch, parse_frames_spec
from dronomy_loc.localize.search import TileCache, search_localize
from dronomy_loc.localize.pipeline import localize_frame
from dronomy_loc.localize.odometry import pairwise_homographies, anchor_from, chain_poses, drift_curve
from dronomy_loc.reference.store import load_reference, save_reference
from dronomy_loc.reference import get_provider
from dronomy_loc.matching import get_matcher
from dronomy_loc.reference.geo import haversine_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    ap.add_argument("--gps-track", default="data/gps_track.csv")
    ap.add_argument("--provider", default="pnoa")
    ap.add_argument("--ref-dir", default="data/reference")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--anchors", type=int, default=28, help="number of anchor keyframes, evenly spread")
    ap.add_argument("--methods", default="sift,loftr", help="matchers to choose among per anchor")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--prior-lat", type=float, default=43.521955)
    ap.add_argument("--prior-lon", type=float, default=-5.624290)
    ap.add_argument("--radius", type=float, default=120.0)
    ap.add_argument("--step", type=float, default=60.0)
    ap.add_argument("--scales", default="60,90,120")
    ap.add_argument("--pixels", type=int, default=640)
    ap.add_argument("--min-inliers", type=int, default=20)
    ap.add_argument("--resize", type=int, default=1280)
    ap.add_argument("--out", default="data/outputs/full_pipeline.csv")
    ap.add_argument("--fig", default="data/outputs/full_flight_path.png")
    args = ap.parse_args()

    cfg = load_config()
    if args.video is None:
        args.video = cfg.video.path
    if hasattr(cfg.matching, "deep"):
        cfg.matching.deep.device = args.device
    track = load_track_csv(args.gps_track)
    n_total = max(f.frame for f in track) + 1
    scales = tuple(float(x) for x in args.scales.split(","))
    anchor_idxs = parse_frames_spec(str(args.anchors), n_total)
    print(f"FULL pipeline: video {n_total} frames · stride {args.stride} · "
          f"{len(anchor_idxs)} anchors · methods={args.methods} · device={args.device}", flush=True)

    # world tile (cached) -> local fetch
    try:
        world = load_reference(args.ref_dir, f"world_{args.provider}")
    except FileNotFoundError:
        world = get_provider(args.provider, cfg).fetch(args.prior_lat, args.prior_lon, 600.0, 4096)
        save_reference(world, args.ref_dir, f"world_{args.provider}")
    fetch = TileCache(make_world_fetch(world))

    # --- Anchor pass: PER-FRAME matcher selection (SIFT vs LoFTR, best lock) ---
    def _mk(name):
        if name == "sift":
            return get_matcher("classical", cfg)
        if name == "roma":                       # native RoMA on the GPU (MPS/CUDA), no Docker
            from dronomy_loc.matching.roma_mps import RomaMpsMatcher
            return RomaMpsMatcher(cfg, device=args.device)
        return get_matcher(name, cfg)
    matchers = {m.strip(): _mk(m.strip()) for m in args.methods.split(",")}
    anchor_frames = grab_frames(args.video, anchor_idxs, resize_long_edge=args.resize)
    anchors, winners = [], {}
    for idx in anchor_idxs:
        best = None
        for name, m in matchers.items():
            t0 = time.time()
            res = search_localize(anchor_frames[idx], args.prior_lat, args.prior_lon, m, fetch,
                                  search_radius_m=args.radius, grid_step_m=args.step,
                                  scales_m=scales, pixels=args.pixels, min_inliers_lock=args.min_inliers)
            if res.locked and (best is None or res.best.n_inliers > best[1].best.n_inliers):
                best = (name, res, m)
        if best is None:
            print(f"anchor {idx}: no matcher locked", flush=True); continue
        name, res, m = best
        tile = fetch(res.best.lat, res.best.lon, res.best.span_m, args.pixels)
        pose, mr = localize_frame(anchor_frames[idx], tile, m)
        if not mr.ok or mr.n_inliers < args.min_inliers:
            print(f"anchor {idx}: re-match below gate", flush=True); continue
        anchors.append(anchor_from(idx, mr.homography, tile))
        winners[name] = winners.get(name, 0) + 1
        gt = gt_for_frame(track, idx)
        err = haversine_m(pose.lat, pose.lon, gt.lat, gt.lon)
        print(f"anchor {idx}: via {name} err={err:.1f}m inliers={mr.n_inliers}", flush=True)
    if not anchors:
        raise SystemExit("no anchors locked")
    print(f"anchors locked: {len(anchors)}/{len(anchor_idxs)} · matcher winners: {winners}", flush=True)

    # --- Sweep pass: SIFT VO over the whole video ---
    vo_cfg = SimpleNamespace(matching=SimpleNamespace(
        classical=SimpleNamespace(detector="SIFT", max_features=3000, ratio_test=0.75),
        ransac=SimpleNamespace(reproj_threshold_px=5.0, confidence=0.999, min_inliers=12)))
    vo = get_matcher("classical", vo_cfg)
    cap = cv2.VideoCapture(str(args.video)); links, prev, shape, idx, n = [], None, None, 0, 0
    t0 = time.time()
    while cap.grab():
        if idx % args.stride == 0:
            ok, img = cap.retrieve()
            if not ok: break
            img = _resize_long_edge(img, args.resize); shape = img.shape; n += 1
            if prev is not None:
                links.append(pairwise_homographies([prev, (idx, img)], vo, min_inliers=12)[0])
            prev = (idx, img)
        idx += 1
    cap.release()
    print(f"sweep: {n} frames, {len(links)} VO links in {time.time()-t0:.0f}s", flush=True)

    # --- Chain VO onto the multi-matcher anchors -> continuous absolute track ---
    chain = chain_poses(links, anchors, shape[:2])
    rows = drift_curve(chain, track)   # frame, est_lat/lon, gt_lat/lon, err_m, hops...
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    errs = sorted(r["err_m"] for r in rows)
    med = errs[len(errs)//2]
    print(f"FULL TRACK: {len(rows)} frames, median err {med:.1f} m, "
          f"anchors {len(anchors)} ({winners})", flush=True)

    # --- One figure: our full flight path vs ground truth ---
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    el=[r["est_lon"] for r in rows]; en=[r["est_lat"] for r in rows]
    gl=[r["gt_lon"] for r in rows]; gn=[r["gt_lat"] for r in rows]
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(gl, gn, "-", color="#1d9e75", lw=3, label="Ground truth (GPS)")
    ax.plot(el, en, "-", color="#d83b2f", lw=1.8, label="Our prediction (full pipeline)")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(f"Full-pipeline flight path vs ground truth\nwhole video · per-frame matcher selection "
                 f"({'+'.join(winners)}) · median {med:.1f} m · GPS-free")
    ax.legend(); ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(args.fig, dpi=140)
    print(f"wrote {args.fig} and {args.out}", flush=True)


if __name__ == "__main__":
    main()
