# Project & code structure

The code is a `pip`-installable Python package (src-layout, package `dronomy_loc`).
Scripts under `scripts/` drive the pipeline step by step; the package holds the
reusable library code they import.

## Directory tree

```
dronomy/                              # git repo root
├── pyproject.toml                    # build config + deps + extras (deep, gee, dev)
├── requirements.txt                  # human-readable dep notes; install via `pip install -e .`
├── README.md                         # usage / APIs (committed)
├── STRUCTURE.md                      # this file (committed)
├── CONTRIBUTING.md                   # contribution rules (committed)
├── PLAN.md / PROJECT_GUIDE.md / ACCURACY_LOG.md   # process docs (committed)
├── .env.example                      # template for keys/paths; real .env is gitignored
├── .gitignore / .dockerignore
├── config/
│   └── config.yaml                   # single source of truth for run-time params
├── src/
│   └── dronomy_loc/                  # the package  ← committed
│       ├── __init__.py
│       ├── config.py                 # typed config dataclasses + YAML loader
│       ├── data/                     # video in, frames + telemetry out
│       │   ├── frames.py             # frame sampling from the drone video
│       │   ├── ingest.py             # sharded, resumable, integrity-checked ingestion
│       │   └── telemetry.py          # exiftool GPS-track extraction (GROUND TRUTH only)
│       ├── reference/                # satellite/orthophoto reference imagery + geo math
│       │   ├── geo.py                # GeoImage, web-mercator bbox, haversine
│       │   ├── base.py               # ReferenceProvider interface
│       │   ├── esri.py               # Esri World Imagery (keyless)
│       │   ├── ign.py                # IGN BD ORTHO WMS, France (keyless)
│       │   ├── pnoa.py               # Spanish IGN PNOA WMS (keyless)
│       │   ├── gee.py                # Google Earth Engine (needs EE_PROJECT auth)
│       │   └── store.py              # local tile cache / store
│       ├── matching/                 # drone<->reference feature matching
│       │   ├── base.py               # Matcher + MatchResult + estimate_homography + get_matcher
│       │   ├── classical.py          # SIFT / ORB matcher
│       │   ├── deep.py               # LoFTR matcher (kornia)
│       │   └── matchanything.py      # MatchAnything backend (Docker-only; imcui)
│       ├── localize/                 # the telemetry-free localization model
│       │   ├── pipeline.py           # PoseEstimate + single-frame localization
│       │   ├── search.py             # coarse-to-fine grid search over the tile
│       │   ├── trajectory.py         # SE(2) alignment + trajectory scoring
│       │   ├── odometry.py           # frame-to-frame VO dead-reckoning + anchors
│       │   └── validate.py           # multi-frame error vs GPS ground truth
│       └── viz/
│           └── overlay.py            # match/overlay/track figures (matplotlib Agg)
├── scripts/                          # CLI entry points, run in numeric order (committed)
│   ├── _bootstrap.py                 # zero-install sys.path shim so scripts run uninstalled
│   ├── 01_extract_frames.py
│   ├── 02_fetch_reference.py
│   ├── 03_localize_frame.py
│   ├── 04_run_video.py
│   ├── 05_ingest_video.py
│   ├── 06_extract_gps_track.py
│   ├── 07_validate.py
│   ├── 08_vo_trajectory.py
│   └── 09_trajectory_report.py
├── tests/                            # pytest unit tests (offline, seeded, committed)
│   ├── test_geo.py
│   ├── test_ingest.py
│   ├── test_matchanything.py
│   ├── test_odometry.py
│   ├── test_providers.py
│   ├── test_search.py
│   ├── test_telemetry.py
│   ├── test_trajectory.py
│   └── test_validate.py
├── docker/
│   ├── Dockerfile                    # dronomy-loc pipeline image (CPU torch + kornia + gee)
│   ├── Dockerfile.matchanything      # dronomy-matchanything image (imcui + real weights)
│   └── docker-compose.yml            # context: repo root; mounts src/scripts/config/data
├── docs/                             # literature review, report, figures (committed)
├── data/                            # local data root; contents gitignored (see below)
├── dronomy_video/                    # the drone video lives here on disk (gitignored)
└── notebooks/                        # exploration (committed shells only)
```

## Install

`pip install -e .` makes `dronomy_loc` importable; add extras as needed:

- `pip install -e ".[deep]"` — LoFTR (CPU torch + kornia).
- `pip install -e ".[gee]"` — Google Earth Engine provider.
- `pip install -e ".[dev]"` — pytest.

Scripts also run uninstalled: `scripts/_bootstrap.py` puts `src/` on `sys.path`,
and `[tool.pytest.ini_options] pythonpath=["src"]` does the same for the tests.
So `python scripts/01_extract_frames.py` and `python -m pytest -q` both work from
a fresh checkout with no install step.

## Run order (pipeline)

| Step | Script                       | What it does                                                      |
|------|------------------------------|-------------------------------------------------------------------|
| 01   | `01_extract_frames.py`       | Sample frames from the drone video into `data/frames/`.           |
| 02   | `02_fetch_reference.py`      | Fetch a georeferenced satellite/orthophoto tile for the area.     |
| 03   | `03_localize_frame.py`       | Localize a single frame -> lat/lon/yaw (the MVP).                 |
| 04   | `04_run_video.py`            | Localize across the video -> trajectory CSV + track-on-map plot.  |
| 05   | `05_ingest_video.py`         | Sharded, resumable, integrity-verified frame ingestion.           |
| 06   | `06_extract_gps_track.py`    | Extract the embedded GPS track (GROUND TRUTH only; never a model input). |
| 07   | `07_validate.py`             | Multi-frame error distribution vs the GPS ground truth.           |
| 08   | `08_vo_trajectory.py`        | Full-trajectory estimate via VO dead-reckoning + absolute anchors.|
| 09   | `09_trajectory_report.py`    | SE(2)-aligned ATE + path-length metrics and the graded figure.    |

```
Video (dronomy_video/*.MP4)
  ├─(01 frames / 05 ingest)─→ data/frames/*.jpg            (drone frames)
  └─(06 extract_gps_track)──→ data/gps_track.csv           (GPS = GROUND TRUTH)

coarse prior + drone frame + reference tile (02)
  └─(03 localize / 04 run_video / 08 vo_trajectory)─→ estimated lat/lon/yaw + trajectory
        │ uses reference.* (tile) + a matcher (matching.classical/.deep/.matchanything)
        └─(07 validate / 09 trajectory_report)─→ error vs GPS ground truth + figures
```

## Shared matcher interface (contract)

Every matcher implements `Matcher.match(...)` and returns a `MatchResult` holding
the corresponding points (Nx2) in both images, the RANSAC homography, the inlier
mask, and the match count — so SIFT, LoFTR, and MatchAnything are directly
comparable and share `estimate_homography` from `matching.base`. `get_matcher`
selects a backend by name from config.

## What is committed vs. not

- **Committed:** `src/dronomy_loc/**`, `scripts/**`, `tests/**`, `docker/**`,
  `config/config.yaml`, `pyproject.toml`, `requirements.txt`, the Markdown docs
  (`README.md`, `STRUCTURE.md`, `CONTRIBUTING.md`, `PLAN.md`, `PROJECT_GUIDE.md`,
  `ACCURACY_LOG.md`, `docs/**`), `.env.example`, `.gitignore`, `.dockerignore`.
- **Never committed (gitignored):** the drone video (`dronomy_video/`, `*.MP4`),
  generated data (`data/frames/`, `data/ingest/`, `data/reference/`,
  `data/outputs/`, `data/cache/`, `data/*.csv`, `data/*.json`), `*.tif`, the
  `venv/`, `.env`, and any credentials/keys (`*.key`, `service-account*.json`).
  The localization model stays telemetry-free — GPS is ground truth only.
