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
  data/        frames.py     Video reading & frame extraction (OpenCV)
  reference/   geo.py        Web-Mercator math + GeoImage (pixel ↔ lat/lon)
               base.py       Provider interface + factory
               ign.py        IGN BD ORTHO via Géoplateforme WMS (open data, no key)
               gee.py        Google Earth Engine provider (stub; needs auth)
               store.py      Save/load fetched reference tiles
  matching/    base.py       Matcher interface + RANSAC homography
               classical.py  SIFT / ORB / AKAZE baseline
               deep.py       LoFTR via kornia (deep matcher)
  localize/    pipeline.py   Homography → (lat, lon, yaw, scale)
  viz/         overlay.py    Match overlays, footprint, trajectory plot
scripts/       01..04        Runnable "small working pieces" (see below)
tests/         test_geo.py   Geo-math sanity checks (no heavy deps)
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

# 2) Fetch a georeferenced satellite tile for the area
python scripts/02_fetch_reference.py --provider ign --span 1500 --pixels 4096

# 3) Localize a single frame (the MVP) — prints lat/lon/yaw, saves overlays
python scripts/03_localize_frame.py --frame data/frames/<one>.jpg --method classical

# 4) Run across the video → trajectory CSV + map plot
python scripts/04_run_video.py --every 2.0 --method classical
```

Run tests: `pytest` (geo math only; no network/torch needed).

## Design notes
- **Reference source is pluggable.** The brief names Google Earth; Adrian sanctioned
  open satellite APIs as a fallback. `provider: ign` (config) gives properly
  *georeferenced* French orthophotos out of the box — ideal for exact pixel↔lat/lon.
- **Matcher is pluggable** so we can satisfy the brief's "compare ≥2 approaches"
  (SIFT vs LoFTR) by changing `--method`.
- **Frames are independent** (per the brief). Temporal smoothing / VO is a later
  extension wired through the same pose output.

## Status & next steps
- [x] Project scaffold, config, geo core, frame extraction, classical + deep matchers
- [ ] Verify IGN tile fetch end-to-end; confirm pixel↔lat/lon round-trip on real data
- [ ] First real single-frame localization (SIFT) + overlay
- [ ] LoFTR path (install torch/kornia) + matcher comparison
- [ ] Full-video trajectory; obtain ground truth for quantitative evaluation
- [ ] Report + presentation; lock meeting with Adrian

See `docs/` for the literature review and report outline.
