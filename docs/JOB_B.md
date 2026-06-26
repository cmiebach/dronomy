# Job B — Models, Evaluation, Output & Runner

**This is the kickoff brief for the teammate's session.** Job A (data &
standardization) is done and on `main`; Job B builds the rest of the framework
against the committed contract. You can start immediately — pull, run `pytest`
(should be green), and build against `dronomy_loc.framework.schema`.

> Prompt to open your session with: **"Implement Job B from docs/JOB_B.md."**

---

## 1. Context (what we're building & how it's graded)

A **generic, plug-and-play framework** that localizes a drone from downward
video across varied terrain — NOT tuned to one video. Per the supervisor
(Adrian): **code quality + framework generality outweigh per-frame accuracy**;
**map-matching is primary, VO secondary**; **RoMA (MatchAnything) is the named
SOTA**; non-urban use case (forests/rivers/rocks/fields, incl. battlefield);
**don't assume a nadir camera**. Deliverables: report (10–20 pg) + A0 poster (due
**29 Jun**), GitHub repo with an **accessible notebook** + ethics statement,
15-min presentation (**6 Jul**).

Three framework components: **Efficiency/Speed**, **Standardization/Consistency**
(Job A — done), **Inversion of Control** (registries + config-driven runner — your
spine).

## 2. The contract you build against (committed by Job A — do NOT modify)

`src/dronomy_loc/framework/schema.py`:
- `CameraIntrinsics(focal_px, principal_point?, dist_coeffs?, hfov_deg?)`
- `Sample(frame_id, image_bgr, t_s?, gt: GPSFix|None, intrinsics?, meta: dict)`
  — `gt` is **scoring only, never a model input** (telemetry-free).
- `Scenario(name, terrain, fetch_tile, sample_iter, prior?, intrinsics?, meta)`
  with `.samples() -> Iterator[Sample]` and `.reference() -> FetchTile`.
- `FetchTile = Callable[[lat, lon, span_m, pixels], GeoImage]` — **the seam.**
  A dataset hands you a `fetch_tile`; the localizer consumes it unchanged.

`src/dronomy_loc/datasets/`: `get_dataset(name, cfg) -> Dataset`, with
`.scenarios() -> list[Scenario]`. Works today for `'video'` (the provided flight)
and `'uavvisloc'` (one Scenario per region). So your runner's input is simply:

```python
from dronomy_loc.datasets import get_dataset
for scenario in get_dataset(name, cfg).scenarios():
    for sample in scenario.samples():
        ... localize sample.image_bgr against scenario.reference() ...
        ... score vs sample.gt ...
```

## 3. What to build (files you own)

| File | Purpose |
|---|---|
| `src/dronomy_loc/models/base.py` | `LocalizationModel` + `get_model(name, cfg)` registry. Wrap `get_matcher` + `search_localize` into a uniform `localize(sample, fetch_tile, cfg) -> FrameScore`. Names: `sift`, `loftr`, **`roma`** (= matchanything model=roma), `eloftr`. |
| `src/dronomy_loc/framework/runner.py` | **The IoC spine.** Config-driven: for each Scenario × model → localize all samples (map-matching primary) → metrics → **select best model per scenario**. Optional VO pass via `odometry` as secondary. |
| `src/dronomy_loc/eval/metrics.py` | Field metrics: haversine error, recall@{1,5,10}m, lock-rate/coverage, trajectory shape ATE (reuse `trajectory.score_trajectory`), runtime; per-terrain/per-dataset aggregation; `select_best(results, metric)`. |
| `src/dronomy_loc/export/geojson.py`, `export/kml.py` | Estimated + GT track in field formats (reuse the existing CSV writer in `validate.py`). |
| `src/dronomy_loc/viz/figures.py` (extend) | Per-model and per-dataset comparison plots. |
| `notebooks/framework_demo.ipynb` | **The graded accessible notebook** — runs the framework on the provided video + one UAV-VisLoc region, shows the metrics table, figures, and a map overlay. |
| `tests/test_models.py`, `test_metrics.py`, `test_export.py` | Offline, synthetic, deterministic. |

## 4. Reuse map (the engine already exists — wrap, don't rewrite)

- `matching/base.py` → `get_matcher('classical'|'loftr'|'matchanything', cfg)`;
  MatchAnything supports `model='roma'`. `estimate_homography` already uses MAGSAC++.
- `localize/search.py` → `search_localize(frame_bgr, prior_lat, prior_lon, matcher, fetch_tile, *, search_radius_m, grid_step_m, scales_m, pixels, min_inliers_lock) -> SearchResult`; `TileCache`.
- `localize/validate.py` → `validate_frames(frames_by_idx, track, prior_lat, prior_lon, matcher, fetch_tile, ...) -> ValidationSummary` (this is ~80% of a per-scenario runner already), `write_validation_csv`, `make_world_fetch`, `grab_frames`, `parse_frames_spec`.
- `localize/trajectory.py` → `score_trajectory` (rigid SE(2)-aligned ATE — the graded shape metric), `align_se2`.
- `localize/odometry.py` → `pairwise_homographies`, `chain_poses`, `drift_curve` (the secondary VO lever).
- `localize/altitude.py` → `estimate_altitude` (brief bonus).
- `viz/figures.py`, `viz/overlay.py` → existing plots to extend.

## 5. ⚠️ Critical finding from real-data testing (the #1 thing to get right)

The provided flight is **~50 m altitude (~71 m footprint)**; UAV-VisLoc region 03
is **~466 m altitude (~840 m footprint)** over an **~8 km** satellite map. The
current search defaults (`radius_m=120`, `scales_m=[50,80,110,140]`) are tuned to
the 50 m flight and **will not lock on UAV-VisLoc**. Your runner must **scale the
search parameters per scene** — derive radius/scales from altitude (`Sample.gt.alt_m`
is GT-only, so prefer `intrinsics` + a coarse altitude prior, or expose per-dataset
search config). This is the main reason a UAV-VisLoc frame only locks sometimes
today. Footprint ≈ `2 * altitude_m * tan(hfov/2)`.

Proven baseline (so you know the path works): a real region-03 frame localized to
**16.7 m** with LoFTR through the contract (GT-centred tile at 840 m span). RoMA +
per-scene search scaling is the accuracy/coverage lever.

## 6. RoMA / MatchAnything

`get_matcher('matchanything', cfg)` with `cfg.matching.matchanything.model='roma'`
is wired but the **real weights run only in `docker/Dockerfile.matchanything`**
(zju3dv fork; pip `imcui` lacks them and pins `numpy<2.3`). For local dev/tests,
mock it exactly like `tests/test_matchanything.py`. Use LoFTR as the runnable
default; RoMA is the SOTA to benchmark in the Docker image.

## 7. Config to add (you own `framework:`)

Add a `framework:` block to `config/config.yaml` (Job A added `camera:` and
`datasets:`): models to benchmark, the selection metric, exporters, and
per-dataset/per-scene search overrides. Keep `[tool.setuptools.packages.find]
where=["src"]` and pytest `pythonpath=["src"]` intact.

## 8. Rules (so the two halves stay decoupled)

- **Do NOT edit** `framework/schema.py`, `datasets/*`, `data/*` (Job A) — import them.
- Don't edit other modules' `__init__.py` mid-build; report the export line and
  integrate centrally (or add your own `models/__init__.py`, `eval/__init__.py`,
  `export/__init__.py`).
- **All tests offline/deterministic** — synthetic scenarios + the provided video;
  **you do NOT need the 2 GB UAV-VisLoc download** to build or test Job B (only to
  run the real UAV-VisLoc benchmark, which one person does once and commits the
  small result CSVs/figures).
- ASCII-only in anything printed at runtime (Windows console).

## 9. Verification targets

- `pytest -q` green, fully offline.
- Runner on the **provided video** → metrics table + figures (LoFTR ~1.7 m on the
  matchable segment).
- Runner on **one UAV-VisLoc region** (with scene-scaled search) → per-region
  metrics; the benchmark table compares SIFT/LoFTR/RoMA and the runner picks the best.
- Notebook executes top-to-bottom and renders metrics + a map overlay.

## 10. Out of scope (note in the report)

DEM/terrain constraints, thermal/IR, full SLAM, active tilt-rectification (design
for tilt *robustness*, don't solve it), distributed compute, plugin discovery.
