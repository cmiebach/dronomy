# dronomy — GPS-denied drone visual localization

Capstone Project · **IE × Dronomy**

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
tests/         test_*.py     56 offline tests: geo, ingest, telemetry, providers, search, validate, VO
docs/                        Literature review + report outline
data/                        Generated artifacts (git-ignored)
```

## Setup
```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -e .                                    # core deps
# Deep matcher (CPU build — no local CUDA):
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

## Status & next steps
- [x] Project scaffold, config, geo core, frame extraction (blur-filtered), classical + deep matchers
- [x] Reference fetch verified end-to-end on real data (Esri + PNOA live; centre round-trip exact)
- [x] Sharded resumable ingestion of the full video (8 shards, 229 frames, integrity-verified)
- [x] GPS ground truth extracted from DJI telemetry (6853 fixes — scoring only, never an input)
- [x] First real single-frame localization (SIFT, grid search): locked on bench frame 6510,
      90.7 m — reproduces the known SIFT baseline/failure-mode on that frame
- [x] Telemetry diagnosis: gimbal nadir all flight, alt ~50 m const (see PLAN.md §0)
- [x] LoFTR bench reproduces caspar's numbers: SIFT 55.3 m (his 56.8), LoFTR 67.7 m
      (his 70.0); frame 6510 locks at **1.76 m** (see ACCURACY_LOG.md)
- [x] Single-scale grid (telemetry-informed): same lock outcome, 1.8× faster
- [x] Validation harness (scripts/07) + ACCURACY_LOG.md
- [x] VO dead-reckoning full-trajectory run (scripts/08): **100% coverage**
      (median 12.3 m, 1.6 m near anchors) vs the ~6% per-frame-matching ceiling
- [ ] Densify anchors; appearance-gap experiments (PLAN.md §3b); 35-stop scan
- [ ] Report + presentation; lock meeting with Adrian

See `PLAN.md` (roadmap), `ACCURACY_LOG.md` (measured results), and
`explained-dronomy.md` (full walkthrough of what/why/how).
