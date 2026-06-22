"""Small working piece #10: regenerate the report figure suite from result files.

Reads the two artifacts already on disk (vo_trajectory.csv from scripts/08 and
bench_results.json from the SIFT-vs-LoFTR sweep) and writes the committed report
PNGs into docs/figures/. It never runs localization or VO — these are pure plots
of numbers we already produced, so they are fast, offline, and reproducible.

Robust on a fresh checkout: if an input file is missing it prints a clear skip
message and moves on (the result files are gitignored) instead of crashing.

Usage:
    python scripts/10_figures.py
    python scripts/10_figures.py --out-dir docs/figures
"""
import argparse

import _bootstrap  # noqa: F401

from dronomy_loc.config import load_config, resolve
from dronomy_loc.viz.figures import (
    fig_bench_bars, fig_coverage, fig_drift_curve, fig_error_vs_frame,
    load_bench_json, load_vo_csv,
)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--vo-csv", default=getattr(cfg.output, "vo_trajectory_csv",
                                                "data/outputs/vo_trajectory.csv"))
    ap.add_argument("--bench-json", default="data/outputs/bench_results.json")
    ap.add_argument("--out-dir", default="docs/figures")
    args = ap.parse_args()

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vo_path = resolve(args.vo_csv)
    bench_path = resolve(args.bench_json)

    # ---- VO figures (need vo_trajectory.csv) ------------------------------
    if vo_path.exists():
        rows = load_vo_csv(vo_path)
        p = fig_drift_curve(rows, out_dir / "drift_curve.png")
        print(f"Wrote {p}  -- VO drift: error vs hops from anchor (the honesty plot)")
        p = fig_error_vs_frame(rows, out_dir / "error_vs_frame.png")
        print(f"Wrote {p}  -- per-frame error along the flight, coloured by hops")
    else:
        print(f"SKIP drift_curve/error_vs_frame: input not found: {vo_path}")

    # ---- Bench figure (needs bench_results.json) --------------------------
    if bench_path.exists():
        bench = load_bench_json(bench_path)
        p = fig_bench_bars(bench, out_dir / "bench_bars.png")
        print(f"Wrote {p}  -- SIFT vs LoFTR per-frame error (hatched = no pose lock)")
    else:
        print(f"SKIP bench_bars: input not found: {bench_path}")

    # ---- Coverage figure (no input file; documented constants) ------------
    p = fig_coverage(out_dir / "coverage.png")
    print(f"Wrote {p}  -- coverage: standalone matching (~6%) vs VO interpolation (100%)")


if __name__ == "__main__":
    main()
