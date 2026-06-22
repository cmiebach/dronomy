"""Visualization helpers: match overlays, trajectory plots, and report figures."""
from .overlay import draw_matches, draw_frame_footprint, plot_trajectory  # noqa: F401
from .figures import (  # noqa: F401
    load_vo_csv, load_bench_json,
    fig_drift_curve, fig_error_vs_frame, fig_bench_bars, fig_coverage,
)
