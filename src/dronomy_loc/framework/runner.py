"""The Inversion-of-Control spine: config in, benchmark out.

The runner owns *no* localization logic. It pulls Scenarios from a dataset
adapter (Job A) and LocalizationModels from the registry (Job B), runs every
model on every scenario through the shared `FetchTile` seam, scores each with
field metrics, and selects the best model per scenario. Adding a dataset or a
model requires zero runner changes — that is the framework's plug-and-play goal.

Per-scene search scaling (the UAV-VisLoc footprint finding) lives here: each
scenario's grid radius/spans are derived from its altitude prior + field of view
(`models.search_for_altitude`), NOT from ground truth. Map-matching is the
primary pass; VO is an optional secondary lever the caller can enable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import islice
from typing import Callable, Iterable

from ..datasets import get_dataset
from ..eval.metrics import FieldMetrics, field_metrics, select_best
from ..localize.validate import FrameScore
from ..models import LocalizationModel, SceneSearch, get_model, search_for_altitude


@dataclass
class ScenarioResult:
    scenario: str
    terrain: str
    dataset: str
    n_samples: int
    per_model: dict[str, FieldMetrics]
    best_model: str | None
    rows_by_model: dict[str, list[FrameScore]] = field(default_factory=dict)


@dataclass
class RunResult:
    select_metric: str
    scenarios: list[ScenarioResult]

    def best_overall(self) -> str | None:
        """Most frequently selected model across scenarios (simple vote)."""
        votes: dict[str, int] = {}
        for sc in self.scenarios:
            if sc.best_model:
                votes[sc.best_model] = votes.get(sc.best_model, 0) + 1
        return max(votes, key=votes.get) if votes else None


def scene_search(scenario, sample0=None) -> SceneSearch:
    """Derive scene-scaled search params from the scenario's altitude prior +
    intrinsics (telemetry-free: altitude prior comes from `scenario.meta`, never
    from `sample.gt`). Falls back to safe defaults when unknown."""
    meta = scenario.meta or {}
    alt = meta.get("altitude_m") or meta.get("altitude_prior_m")
    intr = scenario.intrinsics
    hfov = intr.hfov_deg if intr is not None else None
    focal = intr.focal_px if intr is not None else None
    width = None
    if sample0 is not None and getattr(sample0, "image_bgr", None) is not None:
        try:
            width = sample0.image_bgr.shape[1]
        except Exception:
            width = None
    if alt:
        return search_for_altitude(float(alt), hfov_deg=hfov, focal_px=focal,
                                   image_width_px=width)
    return SceneSearch()


def run_scenario(
    scenario,
    models: dict[str, LocalizationModel],
    *,
    select_metric: str = "recall_5m",
    max_samples: int | None = None,
    on_row: Callable[[str, FrameScore], None] | None = None,
) -> ScenarioResult:
    """Benchmark every model on one scenario and pick the best."""
    if scenario.prior is None:
        raise ValueError(
            f"scenario {scenario.name!r} has no coarse prior; the localizer needs "
            "a rough (lat, lon) to search around (telemetry-free, but not blind)")
    fetch = scenario.reference()
    sample0 = next(iter(scenario.samples()), None)
    search = scene_search(scenario, sample0)
    meta = scenario.meta or {}

    per_model: dict[str, FieldMetrics] = {}
    rows_by_model: dict[str, list[FrameScore]] = {}
    n_samples = 0
    for name, model in models.items():
        rows: list[FrameScore] = []
        for s in islice(scenario.samples(), max_samples):
            row = model.localize(s, fetch, scenario.prior, search=search)
            rows.append(row)
            if on_row is not None:
                on_row(name, row)
        rows_by_model[name] = rows
        per_model[name] = field_metrics(name, rows)
        n_samples = max(n_samples, len(rows))

    return ScenarioResult(
        scenario=scenario.name,
        terrain=scenario.terrain,
        dataset=str(meta.get("dataset", "")),
        n_samples=n_samples,
        per_model=per_model,
        best_model=select_best(per_model, select_metric),
        rows_by_model=rows_by_model,
    )


def run(
    dataset_names: Iterable[str],
    model_names: Iterable[str],
    *,
    cfg=None,
    select_metric: str = "recall_5m",
    max_samples: int | None = None,
    on_row: Callable[[str, FrameScore], None] | None = None,
) -> RunResult:
    """Top-level entry: build models once, iterate datasets x scenarios x models."""
    models = {n: get_model(n, cfg) for n in model_names}
    scenarios: list[ScenarioResult] = []
    for ds_name in dataset_names:
        ds = get_dataset(ds_name, cfg)
        for scenario in ds.scenarios():
            scenarios.append(run_scenario(
                scenario, models, select_metric=select_metric,
                max_samples=max_samples, on_row=on_row))
    return RunResult(select_metric=select_metric, scenarios=scenarios)
