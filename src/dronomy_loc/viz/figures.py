"""Report figures: turn EXISTING result files into the PNGs the brief asks for.

These are pure plotting functions — they take already-parsed data plus an output
Path and write a figure. They never run localization or VO; the whole point is to
visualize what is already on disk (vo_trajectory.csv from scripts/08, bench_results
from the SIFT-vs-LoFTR sweep) so the figures are reproducible offline and stay in
sync with the committed numbers.

The headline is the drift curve: VO is anchored every so often, so error grows with
the number of hops from the last anchor and resets on re-anchor. Showing err_m vs
hops_from_anchor (not vs time) is the honest VO story — it makes the drift-then-reset
sawtooth legible instead of hiding it in a long noisy timeline.

Everything is matplotlib-Agg (headless, writes a PNG) and ASCII-only in any text
drawn on the figure, so it renders the same on a Windows console / CI as elsewhere.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # headless: write a PNG, no display
import matplotlib.pyplot as plt
import numpy as np

# Hop bands for the drift story: fresh (near an anchor), mid, and stale.
HOP_BANDS = [(0, 10, "0-10 hops"), (11, 50, "11-50 hops"), (51, 10 ** 9, "51+ hops")]

# Stable per-config colours so the three figures read consistently.
BENCH_COLORS = {
    "sift_caspar": "#8e8e93",       # grey: the classical baseline
    "loftr_caspar": "#ff9f0a",      # amber: LoFTR on the old grid
    "loftr_ours": "#34c759",        # green: LoFTR, our pipeline
}
BENCH_LABELS = {
    "sift_caspar": "SIFT (baseline)",
    "loftr_caspar": "LoFTR (his grid)",
    "loftr_ours": "LoFTR (ours)",
}


def load_vo_csv(path: str | Path) -> list[dict]:
    """Parse vo_trajectory.csv -> list of row dicts, numeric fields as float
    (None for blanks), sorted by frame. frame/hops_from_anchor/anchor_frame are
    kept as ints for clean axis labels."""
    rows: list[dict] = []
    int_cols = {"frame", "hops_from_anchor", "anchor_frame"}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            row: dict = {}
            for k, v in r.items():
                if v is None or v == "":
                    row[k] = None
                elif k in int_cols:
                    row[k] = int(float(v))
                else:
                    row[k] = float(v)
            rows.append(row)
    rows.sort(key=lambda r: (r.get("frame") if r.get("frame") is not None else 0))
    return rows


def load_bench_json(path: str | Path) -> dict:
    """Parse bench_results.json -> the raw nested dict (config -> frames/mean)."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save(fig, out: str | Path, dpi: int = 130) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def _binned_median(x: np.ndarray, y: np.ndarray, n_bins: int = 24):
    """Median of y within equal-width bins of x; returns (centres, medians) with
    empty bins dropped. Used to draw a trend line through the scatter."""
    if len(x) == 0:
        return np.array([]), np.array([])
    edges = np.linspace(x.min(), x.max() + 1e-9, n_bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    centres, meds = [], []
    for b in range(n_bins):
        sel = y[idx == b]
        if sel.size:
            centres.append(0.5 * (edges[b] + edges[b + 1]))
            meds.append(float(np.median(sel)))
    return np.asarray(centres), np.asarray(meds)


def fig_drift_curve(rows: list[dict], out: str | Path) -> Path:
    """err_m vs hops_from_anchor: scatter + binned-median trend. The headline VO
    honesty plot — error grows with hops since the last anchor, so this shows the
    drift directly. The 0-10 / 11-50 / 51+ hop bands are annotated."""
    pts = [(r["hops_from_anchor"], r["err_m"]) for r in rows
           if r.get("hops_from_anchor") is not None and r.get("err_m") is not None]
    hops = np.array([p[0] for p in pts], float)
    err = np.array([p[1] for p in pts], float)

    fig, ax = plt.subplots(figsize=(10, 6))
    if hops.size:
        ax.scatter(hops, err, s=12, alpha=0.35, color="#0a84ff",
                   edgecolors="none", label="per-frame error")
        cx, cy = _binned_median(hops, err)
        if cx.size:
            ax.plot(cx, cy, "-o", color="#ff3b30", lw=2.0, ms=4,
                    label="binned median")
        ymax = max(err.max(), 1.0)
        hop_max = hops.max()
        band_colors = ["#34c759", "#ff9f0a", "#ff3b30"]
        for (lo, hi, lbl), col in zip(HOP_BANDS, band_colors):
            if lo > hop_max:
                continue
            right = min(hi, hop_max)
            ax.axvspan(lo, right, color=col, alpha=0.07, zorder=0)
            ax.text((lo + right) / 2, ymax * 0.96, lbl, ha="center", va="top",
                    color=col, fontsize=9, weight="bold")
    else:
        ax.text(0.5, 0.5, "no scored rows", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("hops from last anchor")
    ax.set_ylabel("error (m)")
    ax.set_title("VO drift: error grows with hops since the last anchor")
    ax.grid(alpha=0.3)
    if hops.size:
        ax.legend(loc="upper left")
    return _save(fig, out)


def fig_error_vs_frame(rows: list[dict], out: str | Path) -> Path:
    """err_m vs frame index, coloured by hops_from_anchor: shows where the track
    is tight (just re-anchored, dark) vs where it has drifted (many hops, bright).
    The sawtooth = drift then snap-back at each anchor."""
    pts = [(r["frame"], r["err_m"], r["hops_from_anchor"]) for r in rows
           if r.get("frame") is not None and r.get("err_m") is not None
           and r.get("hops_from_anchor") is not None]
    frame = np.array([p[0] for p in pts], float)
    err = np.array([p[1] for p in pts], float)
    hops = np.array([p[2] for p in pts], float)

    fig, ax = plt.subplots(figsize=(11, 6))
    if frame.size:
        ax.plot(frame, err, "-", color="#c7c7cc", lw=0.8, zorder=1)
        sc = ax.scatter(frame, err, c=hops, cmap="viridis", s=16, zorder=2)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("hops from anchor")
    else:
        ax.text(0.5, 0.5, "no scored rows", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("frame index")
    ax.set_ylabel("error (m)")
    ax.set_title("Per-frame error along the flight (colour = hops from anchor)")
    ax.grid(alpha=0.3)
    return _save(fig, out)


def fig_bench_bars(bench: dict, out: str | Path) -> Path:
    """Grouped bar of per-frame error for the three configs (SIFT vs LoFTR his-grid
    vs LoFTR ours). Frames with no pose (err_m null / not locked) get a hatched
    'no lock' marker bar instead of a height, so an unlocked frame is never read as
    zero error. This is the SIFT-vs-LoFTR comparison the brief asks for."""
    configs = [c for c in ("sift_caspar", "loftr_caspar", "loftr_ours") if c in bench]
    # Union of frame ids, ordered numerically where possible.
    frame_ids: list[str] = []
    for c in configs:
        for fid in bench[c].get("frames", {}):
            if fid not in frame_ids:
                frame_ids.append(fid)
    frame_ids.sort(key=lambda s: int(s) if s.isdigit() else s)

    fig, ax = plt.subplots(figsize=(11, 6))
    n_cfg = max(len(configs), 1)
    width = 0.8 / n_cfg
    x = np.arange(len(frame_ids), dtype=float)

    # Reference height for drawing 'no lock' markers (a short hatched stub).
    all_err = [f.get("err_m") for c in configs for f in bench[c]["frames"].values()
               if f.get("err_m") is not None]
    stub = (max(all_err) * 0.04) if all_err else 1.0

    for j, c in enumerate(configs):
        off = (j - (n_cfg - 1) / 2.0) * width
        col = BENCH_COLORS.get(c, "#0a84ff")
        for i, fid in enumerate(frame_ids):
            fr = bench[c].get("frames", {}).get(fid, {})
            e = fr.get("err_m")
            locked = fr.get("locked", e is not None)
            xpos = x[i] + off
            if e is not None and locked:
                ax.bar(xpos, e, width=width, color=col,
                       label=BENCH_LABELS.get(c, c) if i == 0 else None)
                ax.text(xpos, e, f"{e:.1f}", ha="center", va="bottom", fontsize=7)
            else:
                ax.bar(xpos, stub, width=width, color="none", edgecolor=col,
                       hatch="////", linewidth=1.0,
                       label=BENCH_LABELS.get(c, c) if i == 0 else None)
                ax.text(xpos, stub, "no\nlock", ha="center", va="bottom",
                        fontsize=6, color=col)

    ax.set_xticks(x)
    ax.set_xticklabels([f"frame {fid}" for fid in frame_ids])
    ax.set_ylabel("error (m)")
    ax.set_title("Localization error per frame: SIFT vs LoFTR (hatched = no pose lock)")
    ax.grid(alpha=0.3, axis="y")
    if configs:
        ax.legend(loc="upper right")
    # Footnote with the reported means.
    means = "   ".join(
        f"{BENCH_LABELS.get(c, c)}: {bench[c].get('mean_err_m', float('nan')):.1f} m"
        for c in configs)
    if means:
        ax.text(0.0, -0.14, "mean err  " + means, transform=ax.transAxes,
                fontsize=8, color="#3a3a3c")
    return _save(fig, out)


def fig_coverage(out: str | Path, per_frame_pct: float = 6.0,
                 vo_pct: float = 100.0) -> Path:
    """Two-bar coverage story: fraction of frames an absolute matcher can localize
    on its own (about 6 percent, measured) vs frames covered once VO interpolates
    between anchors (100 percent). The numbers are passed in and documented as
    measured so the figure stays honest if they are re-measured."""
    labels = ["Per-frame\nabsolute match", "VO-anchored\ncoverage"]
    vals = [per_frame_pct, vo_pct]
    colors = ["#ff3b30", "#34c759"]

    fig, ax = plt.subplots(figsize=(7, 6))
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=12, weight="bold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("frames localized (%)")
    ax.set_title("Coverage: standalone matching vs VO interpolation")
    ax.grid(alpha=0.3, axis="y")
    ax.text(0.0, -0.12,
            "Numbers measured on this flight; absolute match locks ~%.0f%% of "
            "frames, VO fills the rest." % per_frame_pct,
            transform=ax.transAxes, fontsize=8, color="#3a3a3c")
    return _save(fig, out)


def fig_model_comparison(per_model, out: str | Path, *, title: str = "") -> Path:
    """Benchmark plot for the framework runner: compare models on three field
    KPIs. `per_model` is a mapping {model_name: FieldMetrics} (duck-typed — any
    object with .recall_5m, .lock_rate, .median_err_m). Recall and coverage are
    shown as percentages (higher better); median error as metres (lower better,
    plotted on a twin axis). Missing values render as 0/empty."""
    names = list(per_model)
    recall = [100.0 * getattr(per_model[n], "recall_5m", 0.0) for n in names]
    cover = [100.0 * getattr(per_model[n], "lock_rate", 0.0) for n in names]
    med = [getattr(per_model[n], "median_err_m", None) for n in names]
    x = np.arange(len(names))
    w = 0.38

    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(names) + 4), 6))
    ax.bar(x - w / 2, recall, w, label="recall@5m (%)", color="#0a84ff")
    ax.bar(x + w / 2, cover, w, label="coverage (%)", color="#34c759")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("percent (higher is better)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3, axis="y")

    ax2 = ax.twinx()
    med_plot = [m if m is not None else np.nan for m in med]
    ax2.plot(x, med_plot, "o-", color="#ff3b30", label="median error (m)")
    for xi, m in zip(x, med):
        if m is not None:
            ax2.text(xi, m, f" {m:.1f}m", va="bottom", ha="left",
                     fontsize=9, color="#ff3b30")
    ax2.set_ylabel("median error, m (lower is better)")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", ncol=3, fontsize=9)
    ax.set_title(title or "Model comparison (per scenario)")
    return _save(fig, out)
