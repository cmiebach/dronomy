# dronomy — GPS-denied drone visual localization

Capstone Project · **IE × Dronomy**

> 📊 **All numbers — accuracy, coverage, tests — live in [`STATUS.md`](STATUS.md), the single source of truth.**
> This README covers what the system is, how it's laid out, and how to run it.

Estimate a drone's **absolute pose** (latitude, longitude, and heading) from
**nadir (bottom-looking) video alone**, by matching each frame to a
**georeferenced satellite image** — no GPS, no markers, no environment alteration.

## The challenge
Given a satellite map of the flight area and aerial frames from a drone, estimate
components of the drone pose w.r.t. the map. Pipeline:

```
 drone frame ──► [matching] ──► frame↔reference homography ─┐
 satellite tile (georeferenced) ───────────────────────────┴─► (lat, lon, yaw)
```

Required: (1) auto-fetch a recent satellite image, (2) match frames to a
georeferenced reference, (3) output absolute position/orientation.
Bonus: visual odometry + fusion.

## Run it end-to-end (one command)
A fresh clone needs nothing staged — `run_e2e.py` downloads the flight video,
shards it, extracts the GPS ground truth, fetches reference imagery, then
localizes (each stage is idempotent and independently `--skip`-able):
```bash
pip install -e ".[dev]"
python scripts/run_e2e.py        # fetch video → ingest/shard → GPS → reference → localize
# → data/outputs/run_all/RESULTS.md  (per-method CSVs, auto_track, track.geojson/.kml, figures)
```
Or run just the localizer when the data is already present:
```bash
python scripts/run_all.py --providers pnoa,esri --methods sift,loftr,roma --device cuda
```
LoFTR needs `torch`+`kornia`; RoMA needs a CUDA GPU (see
[`docs/LOCAL_GPU_MATCHANYTHING.md`](docs/LOCAL_GPU_MATCHANYTHING.md)) — both skip
gracefully if their deps are absent. **Numbers: [`STATUS.md`](STATUS.md).**

## Repository layout
```
config/config.yaml          Central configuration (paths, provider, matcher, RANSAC)
src/dronomy_loc/
  data/        frames.py     Video reading & frame extraction, blur filter (OpenCV)
               ingest.py     Sharded, resumable, integrity-verified video ingestion
               telemetry.py  DJI djmd GPS track via exiftool (GROUND TRUTH only)
  reference/   geo.py        Web-Mercator math + GeoImage (pixel ↔ lat/lon)
               base.py       Provider interface + factory
               esri.py       Esri World Imagery (keyless, global, DEFAULT)
               pnoa.py       Spanish IGN PNOA orthophoto (keyless, ~0.15 m/px here)
               gee.py        Google Earth Engine map tiles (needs auth)
               ign.py        French IGN orthophotos (legacy — flight is in Spain)
               store.py      Save/load fetched reference tiles
  matching/    base.py       Matcher interface + RANSAC homography
               classical.py  SIFT / ORB / AKAZE baseline
               deep.py       LoFTR via kornia (deep matcher)
  localize/    pipeline.py   Homography → (lat, lon, yaw, scale)
               search.py     Grid-of-centres × multi-scale search + ≥20-inlier lock gate
               validate.py   Multi-frame validation harness vs the GPS track
               odometry.py   VO dead-reckoning: chain homographies from anchor frames
  viz/         overlay.py    Match overlays, footprint, trajectory plot
scripts/       01..08        Runnable "small working pieces" (see below)
tests/         test_*.py     offline tests: geo, ingest, telemetry, providers, search, validate, VO, fetch
docs/                        Literature review + report outline
data/                        Generated artifacts (git-ignored)
```

## Setup
```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -e .                                    # core deps (any Python 3.10-3.14)
# Deep matcher (CPU build — no local CUDA). NOTE: torch has no wheels for
# Python 3.14 yet — use Python <=3.12 for the loftr/matchanything paths.
# The classical (SIFT) path works on any version. If torch/kornia are missing,
# a --method loftr run now fails fast with a clear message (not "0 inliers").
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install kornia
# Google Earth Engine (optional, primary source per brief): pip install earthengine-api && earthengine authenticate
```

## Quickstart
```bash
# 1) Inspect the video / extract frames
python scripts/01_extract_frames.py --probe
python scripts/01_extract_frames.py --every 2.0 --max 30

# 2) Fetch a georeferenced satellite tile (esri = keyless default; pnoa = best res here)
python scripts/02_fetch_reference.py --provider pnoa --span 500 --pixels 2048

# 3) Localize a single frame (the MVP) — prints lat/lon/yaw, saves overlays
python scripts/03_localize_frame.py --frame data/frames/<one>.jpg --method classical

# 4) Run across the video → trajectory CSV + map plot
python scripts/04_run_video.py --every 2.0 --method classical

# 5) Sharded, resumable ingestion of the whole video (manifest + integrity verify)
python scripts/05_ingest_video.py            # re-run resumes; --verify checks integrity

# 6) Extract the per-frame GPS track (GROUND TRUTH for scoring only; needs exiftool)
python scripts/06_extract_gps_track.py

# 7) Validate N frames against the GPS track → error distribution + CSV
python scripts/07_validate.py --frames 342,3083,6510 --method loftr --provider pnoa

# 8) Full-trajectory VO dead-reckoning anchored on locked frames → drift curve CSV
python scripts/08_vo_trajectory.py --provider pnoa --anchors 6400,6500,6600
```

Run tests: `pytest` (all offline — synthetic videos, mocked network; no torch needed).

## Design notes
- **Reference source is pluggable.** The brief names Google Earth; Adrian sanctioned
  open satellite APIs as a fallback. `provider: esri` (keyless, global) works out of
  the box; `pnoa` (Spanish IGN orthophoto) is the highest-resolution source over the
  true flight area (Asturias) — both give exact pixel↔lat/lon.
- **Matcher is pluggable** so we can satisfy the brief's "compare ≥2 approaches"
  (SIFT vs LoFTR) by changing `--method`.
- **Frames are independent** (per the brief). Temporal smoothing / VO is a later
  extension wired through the same pose output.

## Docs
- **Numbers, results, what's next** → [`STATUS.md`](STATUS.md) — the single source of truth.
- **How it works (long-form walkthrough)** → [`explained-dronomy.md`](explained-dronomy.md).
- **Repo & code structure** → [`STRUCTURE.md`](STRUCTURE.md).
- **Run RoMA on a local GPU** → [`docs/LOCAL_GPU_MATCHANYTHING.md`](docs/LOCAL_GPU_MATCHANYTHING.md).
- **Run RoMA on a cloud GPU pod** → [`docker/RUNPOD.md`](docker/RUNPOD.md).
- **Contributing / conventions** → [`CONTRIBUTING.md`](CONTRIBUTING.md).
