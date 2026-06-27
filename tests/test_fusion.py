"""Recursive fusion filter — offline, deterministic (no I/O, no plotting)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronomy_loc.localize.fusion import (  # noqa: E402
    FusionConfig, FusionStep, fuse_track, fuse_frame_scores,
)
from dronomy_loc.localize.validate import FrameScore  # noqa: E402


def _line_steps(n=40, dt=1.0, speed=5.0, noise=4.0, seed=0):
    """A straight constant-velocity flight with noisy position fixes."""
    rng = np.random.default_rng(seed)
    steps, truth = [], []
    for i in range(n):
        t = i * dt
        ex, ny = speed * t, 0.0          # moving east at `speed`
        truth.append((ex, ny))
        z = (ex + rng.normal(0, noise), ny + rng.normal(0, noise))
        steps.append(FusionStep(t_s=t, pos=z, pos_std=noise))
    return steps, np.array(truth)


def test_smoothing_beats_raw_fixes():
    steps, truth = _line_steps(noise=5.0)
    raw = np.array([s.pos for s in steps])
    est = fuse_track(steps, FusionConfig(fix_std_m=5.0))
    fused = np.array([(e.east, e.north) for e in est])
    raw_rmse = np.sqrt(((raw - truth) ** 2).sum(1).mean())
    fused_rmse = np.sqrt(((fused - truth) ** 2).sum(1).mean())
    assert fused_rmse < raw_rmse            # the filter removes measurement noise


def test_bridges_unlocked_gaps():
    steps, truth = _line_steps(n=30, noise=0.5)
    for i in range(8, 20):                   # a long unlocked stretch
        steps[i] = FusionStep(t_s=steps[i].t_s, pos=None)
    est = fuse_track(steps, FusionConfig(fix_std_m=0.5))
    # The motion model must carry a sensible position through the gap.
    mid = est[14]
    assert abs(mid.east - truth[14, 0]) < 5.0
    assert not mid.fix_used                   # nothing observed there


def test_gate_rejects_outlier_fix():
    steps, _ = _line_steps(n=25, noise=0.3)
    steps[12] = FusionStep(t_s=steps[12].t_s, pos=(steps[12].pos[0] + 200.0, 150.0),
                           pos_std=0.3)        # a 200 m+ wrong lock
    est = fuse_track(steps, FusionConfig(fix_std_m=0.3, gate_chi2=13.816))
    assert est[12].fix_rejected               # the outlier is gated out
    assert abs(est[12].east - 12 * 5.0) < 10.0  # state stays on the true line


def test_disabled_gate_lets_outlier_through():
    steps, _ = _line_steps(n=25, noise=0.3)
    steps[12] = FusionStep(t_s=steps[12].t_s, pos=(steps[12].pos[0] + 200.0, 150.0),
                           pos_std=0.3)
    est = fuse_track(steps, FusionConfig(fix_std_m=0.3, gate_chi2=0.0))
    assert not est[12].fix_rejected           # no gate -> accepted
    assert est[12].north > 20.0               # and it yanks the state off-line


def test_velocity_measurement_constrains_speed():
    # Position-free steps with only a VO velocity should keep the state moving.
    steps = [FusionStep(t_s=0.0, pos=(0.0, 0.0), pos_std=1.0)]
    for i in range(1, 15):
        steps.append(FusionStep(t_s=float(i), vel=(5.0, 0.0), vel_std=0.2))
    est = fuse_track(steps, FusionConfig())
    assert est[-1].v_east == pytest.approx(5.0, abs=1.0)
    assert est[-1].east > 40.0                # carried forward by the velocity


def test_needs_a_position_fix():
    with pytest.raises(ValueError):
        fuse_track([FusionStep(t_s=0.0), FusionStep(t_s=1.0)])


def test_empty_track():
    assert fuse_track([]) == []


def test_causal_filter_runs_without_smoother():
    steps, _ = _line_steps(n=10)
    est = fuse_track(steps, smooth=False)
    assert len(est) == 10


def _frame(idx, est_lat, est_lon, gt_lat, gt_lon, locked, fps=1.0):
    err = None
    return FrameScore(frame=idx, t_s=idx / fps, est_lat=est_lat, est_lon=est_lon,
                      gt_lat=gt_lat, gt_lon=gt_lon, err_m=err, yaw_deg=None,
                      n_inliers=30 if locked else 0, locked=locked, runtime_s=0.1)


def test_fuse_frame_scores_geographic_roundtrip():
    # Drone flies north-east; some frames unlocked, one wildly wrong.
    lat0, lon0 = 43.5219, -5.6243
    rng = np.random.default_rng(3)
    rows = []
    for i in range(30):                   # ~5.5 m/s flight at 1 fps
        gt_lat = lat0 + i * 5e-5
        gt_lon = lon0 + i * 5e-5
        locked = (i % 3 != 0)             # 2/3 of frames lock
        est_lat = gt_lat + rng.normal(0, 5e-5)
        est_lon = gt_lon + rng.normal(0, 5e-5)
        if i == 16:                       # an outlier lock ~330 m away (locked frame)
            est_lat += 3e-3
        rows.append(_frame(i, est_lat if locked else None,
                           est_lon if locked else None, gt_lat, gt_lon, locked))
    fused = fuse_frame_scores(rows, FusionConfig(fix_std_m=6.0))
    assert len(fused) == len(rows)        # every frame gets a position
    # The fused track should track GT to within a few metres at the end.
    last = fused[-1]
    assert abs(last.lat - (lat0 + 29 * 5e-5)) < 1e-4
    assert any(f.fix_rejected for f in fused)   # the i==15 outlier is caught


def test_fuse_frame_scores_needs_a_lock():
    rows = [_frame(0, None, None, 43.5, -5.6, locked=False)]
    with pytest.raises(ValueError):
        fuse_frame_scores(rows)
