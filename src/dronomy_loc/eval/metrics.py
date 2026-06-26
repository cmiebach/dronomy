"""Field metrics for the localization framework.

We score a model on a scenario from its list of per-frame `FrameScore`s (reused
from `localize.validate`, so the contract stays single-sourced). The metrics are
chosen for the *field* use case the supervisor described, not just mean error:

  * **coverage / lock-rate** — what fraction of frames produced a trusted fix
    (a localizer that locks 6% of frames at 1 m is worse in the field than one
    that locks 90% at 8 m);
  * **recall@{1,5,10} m** — fraction of ALL frames that locked AND landed within
    the threshold (rewards coverage and accuracy together, the honest field KPI);
  * **median / mean / worst error** over locked frames (accuracy when it locks);
  * **trajectory-shape ATE** via `trajectory.score_trajectory` (the SE(2)-aligned
    "right shape and dimensions" metric Adrian asked for);
  * **mean runtime** per frame (efficiency).

`select_best` picks the strongest model per scenario by a chosen metric, which is
the framework's "select the best performer per context" requirement.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from ..localize.trajectory import TrajectoryMetrics, score_trajectory
from ..localize.validate import FrameScore

# Metrics where a higher value is better (everything else is lower-is-better).
_HIGHER_IS_BETTER = {"recall_1m", "recall_5m", "recall_10m", "lock_rate"}


@dataclass
class FieldMetrics:
    """Aggregate verdict for one (model, scenario) — or for several scenarios
    combined, if the caller concatenates their rows first."""
    model: str
    n: int                          # total frames scored
    n_locked: int
    lock_rate: float                # coverage in [0, 1]
    median_err_m: float | None      # over LOCKED frames only
    mean_err_m: float | None
    worst_err_m: float | None
    recall_1m: float                # fraction of ALL frames: locked AND err <= 1 m
    recall_5m: float
    recall_10m: float
    mean_runtime_s: float | None
    traj: TrajectoryMetrics | None  # SE(2)-aligned trajectory shape (None if < 2 locked fixes)


def recall_at(rows: list[FrameScore], threshold_m: float) -> float:
    """Fraction of ALL frames that locked and landed within `threshold_m`.

    Denominator is every frame (not just locked ones) on purpose: this couples
    coverage and accuracy into one field KPI, so a model cannot win by locking
    one easy frame perfectly and skipping the rest.
    """
    n = len(rows)
    if n == 0:
        return 0.0
    hit = sum(1 for r in rows
              if r.locked and r.err_m is not None and r.err_m <= threshold_m)
    return hit / n


def field_metrics(model: str, rows: list[FrameScore]) -> FieldMetrics:
    """Aggregate a model's per-frame scores on a scenario into FieldMetrics."""
    n = len(rows)
    locked = [r for r in rows if r.locked and r.err_m is not None]
    errs = [r.err_m for r in locked]
    runtimes = [r.runtime_s for r in rows if r.runtime_s is not None]

    # Trajectory shape over locked frames that have an estimate (need >= 2 points).
    pairs = [r for r in locked if r.est_lat is not None and r.est_lon is not None]
    traj = None
    if len(pairs) >= 2:
        traj = score_trajectory(
            [r.est_lat for r in pairs], [r.est_lon for r in pairs],
            [r.gt_lat for r in pairs], [r.gt_lon for r in pairs],
        )

    return FieldMetrics(
        model=model,
        n=n,
        n_locked=len(locked),
        lock_rate=(len(locked) / n) if n else 0.0,
        median_err_m=statistics.median(errs) if errs else None,
        mean_err_m=statistics.fmean(errs) if errs else None,
        worst_err_m=max(errs) if errs else None,
        recall_1m=recall_at(rows, 1.0),
        recall_5m=recall_at(rows, 5.0),
        recall_10m=recall_at(rows, 10.0),
        mean_runtime_s=statistics.fmean(runtimes) if runtimes else None,
        traj=traj,
    )


def select_best(results: dict[str, FieldMetrics], metric: str = "recall_5m") -> str | None:
    """Return the model name that scores best on `metric` (None if undecidable).

    Higher-is-better for recall_*/lock_rate; lower-is-better for the error
    metrics and `ate_aligned_m` (read off `traj`). Models for which the metric is
    undefined (e.g. median error with nothing locked) are skipped.
    """
    def value(m: FieldMetrics):
        if metric == "ate_aligned_m":
            return m.traj.ate_aligned_m if m.traj is not None else None
        return getattr(m, metric, None)

    scored = [(name, value(m)) for name, m in results.items()]
    scored = [(name, v) for name, v in scored if v is not None]
    if not scored:
        return None
    higher = metric in _HIGHER_IS_BETTER
    pick = max if higher else min
    return pick(scored, key=lambda kv: kv[1])[0]
