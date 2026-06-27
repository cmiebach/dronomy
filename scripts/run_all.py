"""Single end-to-end entrypoint — one trigger, complete outputs.

This is the command an evaluator runs. It localizes a set of frames with EVERY
matcher (SIFT, LoFTR, RoMA), scores each against GPS ground truth, lets the
framework pick the best matcher PER FRAME by lock confidence (the auto-selection
that is the project's point), and writes the full output set:

  <out>/val_<method>.csv      per-method per-frame results
  <out>/auto_track.csv        the auto-selected (best-per-frame) trajectory
  <out>/track.geojson / .kml  the auto-selected track, for any GIS/Earth viewer
  <out>/comparison.png        per-method coverage/accuracy bars
  <out>/flightpath.png        GT vs auto-selected predicted path
  <out>/RESULTS.md            metrics table + framework pick + cascade coverage

RoMA needs the MatchAnything deps (the Docker image / GPU pod). Where they are
present it runs in-process (fast on CUDA, slow emulated); where they are absent
the run does NOT crash — RoMA is reported as unavailable and the other methods
still produce complete outputs. Telemetry-free: GPS is scoring only.

Examples:
  python scripts/run_all.py --frames-dir data/roma_frames --device cuda     # pod/GPU
  python scripts/run_all.py --video <mp4> --spread 20 --methods sift,loftr   # host
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronomy_loc.config import load_config           # noqa: E402
from dronomy_loc.data.telemetry import load_track_csv  # noqa: E402
from dronomy_loc.localize.validate import (           # noqa: E402
    FrameScore, grab_frames, make_world_fetch, parse_frames_spec,
    validate_frames, write_validation_csv, ValidationSummary,
)
from dronomy_loc.localize.search import TileCache     # noqa: E402
from dronomy_loc.reference.store import load_reference  # noqa: E402
from dronomy_loc.reference.geo import haversine_m      # noqa: E402
from dronomy_loc.eval.metrics import field_metrics, select_best  # noqa: E402


def build_matcher(method, cfg, device):
    """Registry with RoMA device override; raises if a method's deps are absent."""
    m = method.lower()
    if m in ("sift", "classical"):
        from dronomy_loc.matching import get_matcher
        return get_matcher("classical", cfg)
    if m == "loftr":
        from dronomy_loc.matching import get_matcher
        return get_matcher("loftr", cfg)
    if m in ("roma", "eloftr", "matchanything"):
        from dronomy_loc.matching.matchanything import MatchAnythingMatcher
        mm = MatchAnythingMatcher(cfg, model="roma" if m in ("roma", "matchanything") else "eloftr")
        mm.device = device
        return mm
    raise ValueError(f"unknown method {method!r}")


def load_frames(args, indices):
    if args.frames_dir:
        import cv2
        out = {}
        for idx in indices:
            p = Path(args.frames_dir) / f"frame_{idx}.jpg"
            img = cv2.imread(str(p))
            if img is not None:
                out[idx] = img
        return out
    return grab_frames(args.video, indices)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="dronomy_video/IE_Challenge_lat43_521955_lon5_624290.MP4")
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--gps-track", default="data/gps_track.csv")
    ap.add_argument("--ref-dir", default="data/reference")
    ap.add_argument("--ref-name", default="world_pnoa")
    ap.add_argument("--providers", default=None,
                    help="comma list (e.g. pnoa,esri) -> per-frame best-source selection")
    ap.add_argument("--spread", type=int, default=20)
    ap.add_argument("--methods", default="sift,loftr,roma")
    ap.add_argument("--device", default="cpu", help="RoMA backend: cpu | cuda | mps")
    ap.add_argument("--prior-lat", type=float, default=43.521955)
    ap.add_argument("--prior-lon", type=float, default=-5.624290)
    ap.add_argument("--radius", type=float, default=100.0)
    ap.add_argument("--step", type=float, default=50.0)
    ap.add_argument("--scales", default="80,110,140")
    ap.add_argument("--pixels", type=int, default=640)
    ap.add_argument("--min-inliers", type=int, default=20)
    ap.add_argument("--fps", type=float, default=29.97)
    ap.add_argument("--out-dir", default="data/outputs/run_all")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    track = load_track_csv(args.gps_track)
    n_total = max(f.frame for f in track) + 1
    indices = parse_frames_spec(str(args.spread), n_total)
    scales = tuple(float(x) for x in args.scales.split(","))
    print(f"run_all: {len(indices)} frames x methods={args.methods} (device={args.device})", flush=True)

    frames = load_frames(args, indices)
    if not frames:
        raise SystemExit("no frames loaded (check --frames-dir / --video)")
    # one or several imagery providers (multi-source = per-frame best-source selection)
    prov_names = [p.strip() for p in args.providers.split(",")] if args.providers else [None]
    providers = {}
    for p in prov_names:
        name = p or args.ref_name.replace("world_", "")
        cache = p and f"world_{p}" or args.ref_name
        providers[name] = TileCache(make_world_fetch(load_reference(args.ref_dir, cache)))
    multi = len(providers) > 1
    fetch_tile = next(iter(providers.values()))
    if multi:
        from dronomy_loc.localize.multisource import validate_multisource
        print(f"multi-source providers: {list(providers)}", flush=True)

    # --- run every method; a missing-deps method is skipped, not fatal ---
    summaries: dict[str, ValidationSummary] = {}
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        print(f"\n=== {method} ===", flush=True)
        try:
            matcher = build_matcher(method, cfg, args.device)
            if multi:
                summ, _ = validate_multisource(
                    frames, track, args.prior_lat, args.prior_lon, matcher, providers,
                    fps=args.fps, search_radius_m=args.radius, grid_step_m=args.step,
                    scales_m=scales, pixels=args.pixels, min_inliers_lock=args.min_inliers,
                    on_row=lambda r, name: print(f"  frame {r.frame} locked={int(r.locked)} "
                                                 f"via={name} inl={r.n_inliers} "
                                                 f"err={'' if r.err_m is None else round(r.err_m,1)}", flush=True))
            else:
                summ = validate_frames(
                    frames, track, args.prior_lat, args.prior_lon, matcher, fetch_tile,
                    fps=args.fps, search_radius_m=args.radius, grid_step_m=args.step,
                    scales_m=scales, pixels=args.pixels, min_inliers_lock=args.min_inliers,
                    on_row=lambda r: print(f"  frame {r.frame} locked={int(r.locked)} "
                                           f"inl={r.n_inliers} err={'' if r.err_m is None else round(r.err_m,1)}",
                                           flush=True))
            summaries[method] = summ
            write_validation_csv(summ, out / f"val_{method}.csv")
        except Exception as e:               # missing torch/imcui/romatch etc.
            print(f"  {method} UNAVAILABLE in this environment: {e}", flush=True)

    if not summaries:
        raise SystemExit("no method produced results")

    # --- auto-selection: per frame, trust the highest-confidence locked method ---
    by_frame: dict[int, list[tuple[str, FrameScore]]] = {}
    for name, summ in summaries.items():
        for r in summ.rows:
            by_frame.setdefault(r.frame, []).append((name, r))
    auto_rows, winners = [], {}
    for f in sorted(by_frame):
        locked = [(n, r) for n, r in by_frame[f] if r.locked and r.est_lat is not None]
        if not locked:
            continue
        n, r = max(locked, key=lambda nr: nr[1].n_inliers)   # confidence = inliers
        winners[n] = winners.get(n, 0) + 1
        auto_rows.append(r)
    auto_summary = ValidationSummary(
        n=len(by_frame), n_locked=len(auto_rows),
        lock_rate=len(auto_rows) / len(by_frame) if by_frame else 0.0,
        median_err_m=statistics.median([r.err_m for r in auto_rows if r.err_m is not None]) if auto_rows else None,
        mean_err_m=statistics.fmean([r.err_m for r in auto_rows if r.err_m is not None]) if auto_rows else None,
        worst_err_m=max([r.err_m for r in auto_rows if r.err_m is not None]) if auto_rows else None,
        rows=auto_rows)
    write_validation_csv(auto_summary, out / "auto_track.csv")

    # --- exports + figures ---
    from dronomy_loc.export.geojson import write_geojson
    from dronomy_loc.export.kml import write_kml
    write_geojson(auto_rows, out / "track.geojson", name="auto-selected")
    write_kml(auto_rows, out / "track.kml", name="auto-selected")

    per_model = {name: field_metrics(name, s.rows) for name, s in summaries.items()}
    best = select_best(per_model, "recall_5m")
    try:
        from dronomy_loc.viz.figures import fig_model_comparison
        fig_model_comparison(per_model, out / "comparison.png", title="Matcher comparison")
    except Exception as e:
        print(f"comparison figure skipped: {e}", flush=True)
    _flightpath_png(auto_rows, out / "flightpath.png")

    # --- RESULTS.md ---
    L = ["# End-to-end run — all methods (single trigger)", "",
         f"Frames: {len(indices)} · methods run: {', '.join(summaries)} · device: {args.device}", "",
         "| Method | Coverage | recall@5m | median err | mean err |",
         "|---|---|---|---|---|"]
    for name, fm in per_model.items():
        med = f"{fm.median_err_m:.1f} m" if fm.median_err_m is not None else "—"
        mean = f"{fm.mean_err_m:.1f} m" if fm.mean_err_m is not None else "—"
        L.append(f"| {name} | {fm.lock_rate*100:.0f}% | {fm.recall_5m:.2f} | {med} | {mean} |")
    L += ["", f"**Framework pick (recall@5m): `{best}`**", "",
          f"**Auto-selected (best-confidence per frame): coverage "
          f"{auto_summary.lock_rate*100:.0f}%, median err "
          f"{auto_summary.median_err_m:.1f} m**" if auto_summary.median_err_m is not None else "",
          f"Per-frame winners: " + ", ".join(f"{k} {v}" for k, v in sorted(winners.items())),
          "", "Outputs: val_<method>.csv, auto_track.csv, track.geojson/.kml, "
          "comparison.png, flightpath.png"]
    missing = set(m.strip() for m in args.methods.split(",")) - set(summaries)
    if missing:
        L += ["", f"_Methods unavailable in this environment: {', '.join(sorted(missing))} "
              "(need the MatchAnything Docker image / GPU pod for RoMA)._"]
    (out / "RESULTS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nAll outputs in {out}/")


def _flightpath_png(rows, path):
    """GT (green) vs auto-selected predicted (red) in lon/lat."""
    if not rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows = sorted(rows, key=lambda r: r.frame)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([r.gt_lon for r in rows], [r.gt_lat for r in rows], "-o", color="#1d9e75",
            ms=4, lw=1.5, label="ground truth (GPS)")
    ax.plot([r.est_lon for r in rows], [r.est_lat for r in rows], "x", color="#d85a30",
            ms=7, mew=2, label="predicted (auto-selected)")
    for r in rows:
        ax.plot([r.gt_lon, r.est_lon], [r.gt_lat, r.est_lat], "-", color="#cccccc", lw=0.6, zorder=0)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title("Auto-selected localization vs ground truth")
    ax.legend(); ax.set_aspect("equal", adjustable="datalim"); fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
