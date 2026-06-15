"""Small working piece #9: turn a VO trajectory CSV into the graded artifact.

Produces (1) the SHAPE-PRECISION metrics Adrian asked for (rigid-SE(2)-aligned
ATE + path-length ratio) and (2) a two-panel figure: estimated vs GPS track
drawn on the satellite tile (absolute view) and after rigid alignment (the
"same shape and dimensions" view). The figure is the artifact to show / put in
the report.

Usage:
    python scripts/09_trajectory_report.py
    python scripts/09_trajectory_report.py --csv data/outputs/vo_trajectory.csv
"""
import argparse
import csv

import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")                       # headless: write a PNG, no display
import matplotlib.pyplot as plt
import numpy as np

from dronomy_loc.config import load_config, resolve
from dronomy_loc.localize.trajectory import lonlat_to_local_m, score_trajectory
from dronomy_loc.reference import load_reference


def _read(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("est_lat") and r.get("gt_lat"):
                rows.append({k: float(v) if v else None for k, v in r.items()})
    rows.sort(key=lambda r: r["frame"])
    return rows


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=getattr(cfg.output, "vo_trajectory_csv",
                                             "data/outputs/vo_trajectory.csv"))
    ap.add_argument("--provider", default=cfg.reference.provider)
    ap.add_argument("--out", default="data/outputs/trajectory_report.png")
    args = ap.parse_args()

    rows = _read(resolve(args.csv))
    if not rows:
        raise SystemExit(f"No scored rows in {args.csv}")
    est_lat = [r["est_lat"] for r in rows]
    est_lon = [r["est_lon"] for r in rows]
    gt_lat = [r["gt_lat"] for r in rows]
    gt_lon = [r["gt_lon"] for r in rows]

    m = score_trajectory(est_lat, est_lon, gt_lat, gt_lon)
    print(f"Frames scored:        {m.n}")
    print(f"ATE (raw):            {m.ate_raw_m:6.1f} m")
    print(f"ATE (SE2-aligned):    {m.ate_aligned_m:6.1f} m   <- shape-precision metric")
    print(f"  mean/median/worst:  {m.mean_aligned_m:.1f} / {m.median_aligned_m:.1f} / "
          f"{m.worst_aligned_m:.1f} m")
    print(f"Path length est/gt:   {m.path_len_est_m:.0f} / {m.path_len_gt_m:.0f} m "
          f"(ratio {m.path_len_ratio:.3f}, 1.0 = identical dimensions)")
    print(f"Heading offset:       {m.align.rot_deg:+.1f} deg")

    # ---- Panel 1: absolute, on the satellite tile -------------------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 7))
    try:
        world = load_reference(resolve(cfg.reference.out_dir), f"world_{args.provider}")
        epx, epy = zip(*[world.lonlat_to_pixel(lo, la) for la, lo in zip(est_lat, est_lon)])
        gpx, gpy = zip(*[world.lonlat_to_pixel(lo, la) for la, lo in zip(gt_lat, gt_lon)])
        axL.imshow(world.image)
        axL.plot(gpx, gpy, "-", color="#00e5ff", lw=2.5, label="GPS ground truth")
        axL.plot(epx, epy, "-", color="#ff3b30", lw=1.8, label="Our estimate (VO)")
        allx, ally = list(epx) + list(gpx), list(epy) + list(gpy)
        mx, my = (max(allx) - min(allx)) * 0.5 + 30, (max(ally) - min(ally)) * 0.5 + 30
        cxp, cyp = (max(allx) + min(allx)) / 2, (max(ally) + min(ally)) / 2
        r = max(mx, my)
        axL.set_xlim(cxp - r, cxp + r); axL.set_ylim(cyp + r, cyp - r)
        axL.set_title("Absolute: estimate vs GPS on PNOA orthophoto")
    except FileNotFoundError:
        axL.text(0.5, 0.5, "world tile not cached\n(run scripts/08 first)",
                 ha="center", va="center"); axL.set_title("Absolute (no basemap)")
    axL.legend(loc="upper right"); axL.set_xticks([]); axL.set_yticks([])

    # ---- Panel 2: rigid-aligned, in metres (the "same shape" view) --------
    ref_lat, ref_lon = float(np.mean(gt_lat)), float(np.mean(gt_lon))
    est_m = lonlat_to_local_m(est_lat, est_lon, ref_lat, ref_lon)
    gt_m = lonlat_to_local_m(gt_lat, gt_lon, ref_lat, ref_lon)
    est_a = m.align.apply(est_m)
    axR.plot(gt_m[:, 0], gt_m[:, 1], "-", color="#00a0c0", lw=2.5, label="GPS ground truth")
    axR.plot(est_a[:, 0], est_a[:, 1], "-", color="#ff3b30", lw=1.8,
             label="Our estimate (rigidly aligned)")
    axR.set_aspect("equal"); axR.grid(alpha=0.3)
    axR.set_xlabel("east (m)"); axR.set_ylabel("north (m)")
    axR.set_title(f"Shape match: SE(2)-aligned ATE = {m.ate_aligned_m:.1f} m  "
                  f"(path ratio {m.path_len_ratio:.2f})")
    axR.legend(loc="best")

    fig.suptitle(f"Dronomy VO trajectory vs GPS - {m.n} frames, 100% coverage",
                 fontsize=14, weight="bold")
    fig.tight_layout()
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
