"""Run localization across the whole video (frames treated independently, per the
brief) and write a trajectory CSV + a track-on-map plot.

Usage:
    python scripts/04_run_video.py --every 2.0 --method classical
"""
import argparse
import csv

import _bootstrap  # noqa: F401
from tqdm import tqdm

from dronomy_loc.config import load_config, resolve
from dronomy_loc.data import frames as frames_mod
from dronomy_loc.matching import get_matcher
from dronomy_loc.localize import localize_frame
from dronomy_loc.viz import plot_trajectory
from dronomy_loc.reference import load_reference


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=float, default=cfg.frames.every_n_seconds)
    ap.add_argument("--max", type=int, default=cfg.frames.max_frames)
    ap.add_argument("--method", default=cfg.matching.method)
    ap.add_argument("--provider", default=cfg.reference.provider)
    args = ap.parse_args()

    ref = load_reference(resolve(cfg.reference.out_dir), args.provider)
    matcher = get_matcher(args.method, cfg)
    video = resolve(cfg.video.path)

    rows, lats, lons = [], [], []
    for fi in tqdm(frames_mod.iter_frames(
            video, every_n_seconds=args.every, max_frames=args.max,
            resize_long_edge=cfg.frames.resize_long_edge), desc="localizing"):
        pose, mr = localize_frame(fi.image, ref, matcher)
        rows.append({
            "frame": fi.index, "t_s": round(fi.t_seconds, 3),
            "ok": pose.ok, "lat": pose.lat, "lon": pose.lon,
            "yaw_deg": pose.yaw_deg, "ground_m_per_px": pose.ground_m_per_px,
            "n_inliers": pose.n_inliers, "n_matches": pose.n_matches,
        })
        if pose.ok:
            lats.append(pose.lat); lons.append(pose.lon)

    out_csv = resolve(cfg.output.trajectory_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} rows ({len(lats)} localized) -> {out_csv}")

    if lats:
        plot = resolve(cfg.output.dir) / f"trajectory_{args.method}.png"
        plot_trajectory(ref, lats, lons, plot, title=f"Trajectory ({args.method})")
        print(f"Saved {plot}")


if __name__ == "__main__":
    main()
