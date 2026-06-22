"""Report figure generators - offline, deterministic, synthetic data only.

These never touch the real result files (they are gitignored). Each generator is
fed small in-memory rows/bench dicts and must write a non-trivial PNG. We also
round-trip the two loaders through a tiny written fixture.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronomy_loc.viz.figures import (  # noqa: E402
    fig_bench_bars, fig_coverage, fig_drift_curve, fig_error_vs_frame,
    load_bench_json, load_vo_csv,
)

PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _assert_png(path: Path):
    assert isinstance(path, Path)
    assert path.exists()
    data = path.read_bytes()
    assert data[:8] == PNG_SIG          # real PNG signature
    assert len(data) > 1024             # non-trivial


def _synth_rows(n=60):
    # Sawtooth hops: drift then re-anchor; err grows with hops.
    rows = []
    for i in range(n):
        hops = i % 20                    # 0..19 repeating
        anchor = (i // 20) * 200
        rows.append({
            "frame": i * 10,
            "hops_from_anchor": hops,
            "anchor_frame": anchor,
            "err_m": 2.0 + hops * 3.0,    # monotonic in hops
            "est_lat": 43.522 + i * 1e-5,
            "est_lon": -5.624 + i * 1e-5,
            "gt_lat": 43.521 + i * 1e-5,
            "gt_lon": -5.623 + i * 1e-5,
        })
    # include a couple of stale (51+) hop rows for the third band
    rows.append({"frame": 9000, "hops_from_anchor": 80, "anchor_frame": 8000,
                 "err_m": 150.0, "est_lat": 43.5, "est_lon": -5.6,
                 "gt_lat": 43.5, "gt_lon": -5.6})
    return rows


def _synth_bench():
    return {
        "sift_caspar": {
            "frames": {
                "342": {"err_m": 19.97, "n_inliers": 39, "locked": True},
                "3083": {"err_m": None, "n_inliers": 0, "locked": False},
                "6510": {"err_m": 90.73, "n_inliers": 133, "locked": True},
            },
            "mean_err_m": 55.3,
        },
        "loftr_caspar": {
            "frames": {
                "342": {"err_m": 106.67, "n_inliers": 15, "locked": False},
                "3083": {"err_m": 94.68, "n_inliers": 15, "locked": False},
                "6510": {"err_m": 1.76, "n_inliers": 117, "locked": True},
            },
            "mean_err_m": 67.7,
        },
        "loftr_ours": {
            "frames": {
                "342": {"err_m": None, "n_inliers": 0, "locked": False},
                "3083": {"err_m": None, "n_inliers": 0, "locked": False},
                "6510": {"err_m": 1.75, "n_inliers": 35, "locked": True},
            },
            "mean_err_m": 1.7,
        },
    }


def test_fig_drift_curve(tmp_path):
    out = fig_drift_curve(_synth_rows(), tmp_path / "drift.png")
    _assert_png(out)


def test_fig_error_vs_frame(tmp_path):
    out = fig_error_vs_frame(_synth_rows(), tmp_path / "evf.png")
    _assert_png(out)


def test_fig_bench_bars(tmp_path):
    out = fig_bench_bars(_synth_bench(), tmp_path / "bench.png")
    _assert_png(out)


def test_fig_coverage(tmp_path):
    out = fig_coverage(tmp_path / "cov.png")
    _assert_png(out)


def test_fig_coverage_custom_numbers(tmp_path):
    out = fig_coverage(tmp_path / "cov2.png", per_frame_pct=4.0, vo_pct=95.0)
    _assert_png(out)


def test_generators_handle_empty(tmp_path):
    # No scored rows / no configs must still produce a valid PNG (no crash).
    _assert_png(fig_drift_curve([], tmp_path / "d0.png"))
    _assert_png(fig_error_vs_frame([], tmp_path / "e0.png"))
    _assert_png(fig_bench_bars({}, tmp_path / "b0.png"))


def test_load_vo_csv(tmp_path):
    p = tmp_path / "vo.csv"
    p.write_text(
        "frame,hops_from_anchor,anchor_frame,err_m,est_lat,est_lon,gt_lat,gt_lon\n"
        "20,2,6400,5.5,43.5,-5.6,43.4,-5.5\n"
        "0,640,6400,68.51,43.52,-5.62,43.52,-5.62\n"
        "10,,6400,,43.5,-5.6,43.4,-5.5\n",          # blank hops + err -> None
        encoding="utf-8",
    )
    rows = load_vo_csv(p)
    assert [r["frame"] for r in rows] == [0, 10, 20]   # sorted by frame
    assert rows[0]["hops_from_anchor"] == 640 and isinstance(rows[0]["frame"], int)
    assert rows[1]["hops_from_anchor"] is None and rows[1]["err_m"] is None
    assert rows[2]["err_m"] == 5.5


def test_load_bench_json(tmp_path):
    p = tmp_path / "bench.json"
    p.write_text(json.dumps(_synth_bench()), encoding="utf-8")
    bench = load_bench_json(p)
    assert set(bench) == {"sift_caspar", "loftr_caspar", "loftr_ours"}
    assert bench["loftr_ours"]["mean_err_m"] == 1.7
    assert bench["sift_caspar"]["frames"]["3083"]["err_m"] is None
