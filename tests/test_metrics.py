"""Field metrics: coverage, recall@t, error stats, trajectory shape, selection.
Offline + deterministic (synthetic FrameScores)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronomy_loc.eval.metrics import field_metrics, recall_at, select_best  # noqa: E402
from dronomy_loc.localize.validate import FrameScore  # noqa: E402

LAT, LON = 43.5220, -5.6243
_D = 1.0 / 111_320.0  # ~1 m in latitude degrees


def _row(frame, err_m, locked, dn_m=0.0, runtime=0.1):
    """A FrameScore whose estimate sits dn_m metres north of a moving GT point
    (so a sequence forms a real trajectory)."""
    gt_lat = LAT + frame * 5 * _D     # GT walks north 5 m/frame
    return FrameScore(
        frame=frame, t_s=float(frame),
        est_lat=(gt_lat + dn_m * _D) if locked else None,
        est_lon=LON if locked else None,
        gt_lat=gt_lat, gt_lon=LON,
        err_m=err_m, yaw_deg=12.0 if locked else None,
        n_inliers=50 if locked else 3, locked=locked, runtime_s=runtime,
    )


ROWS = [
    _row(0, 0.5, True, dn_m=0.5),
    _row(1, 3.0, True, dn_m=3.0),
    _row(2, 8.0, True, dn_m=8.0),
    _row(3, None, False),
]


def test_recall_at_couples_coverage_and_accuracy():
    assert recall_at(ROWS, 1.0) == 0.25    # only the 0.5 m frame, over all 4
    assert recall_at(ROWS, 5.0) == 0.50    # 0.5 + 3.0
    assert recall_at(ROWS, 10.0) == 0.75   # 0.5 + 3.0 + 8.0
    assert recall_at([], 5.0) == 0.0


def test_field_metrics_aggregate():
    m = field_metrics("loftr", ROWS)
    assert m.n == 4 and m.n_locked == 3
    assert m.lock_rate == 0.75
    assert m.median_err_m == 3.0
    assert m.worst_err_m == 8.0
    assert abs(m.mean_runtime_s - 0.1) < 1e-9
    assert m.traj is not None and m.traj.n == 3   # 3 locked points -> shape metric


def test_field_metrics_nothing_locked_is_none_not_zero():
    m = field_metrics("sift", [_row(0, None, False), _row(1, None, False)])
    assert m.n_locked == 0 and m.lock_rate == 0.0
    assert m.median_err_m is None and m.mean_err_m is None and m.worst_err_m is None
    assert m.traj is None


def test_select_best_direction():
    a = field_metrics("loftr", ROWS)                       # recall_5m 0.5
    b = field_metrics("sift", [_row(0, 9.0, True, dn_m=9.0),
                               _row(1, 9.0, True, dn_m=9.0)])  # recall_5m 0.0, but median 9
    res = {"loftr": a, "sift": b}
    assert select_best(res, "recall_5m") == "loftr"        # higher is better
    assert select_best(res, "median_err_m") == "loftr"     # 3.0 < 9.0, lower better
    assert select_best({}, "recall_5m") is None
