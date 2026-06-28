# Project status, single source of truth for numbers

**Last updated 2026-06-29.** Use these numbers everywhere. They match the report and the poster. If a figure is not here, ask before putting it on the poster or in the report.

## 1. What the system is
A telemetry free, modular framework that localizes a drone from its downward camera alone, by matching each video frame to georeferenced satellite imagery. No GPS at runtime, GPS is used only to score. Matchers (SIFT, LoFTR, RoMA) and imagery providers (PNOA, Esri, Google, GEE) are interchangeable behind one interface, and the same pipeline runs unchanged across terrains and datasets.

## 2. Headline numbers (confirmed, safe to use)

| Capability | Result | How measured |
|---|---|---|
| **RoMA blind whole video pipeline (headline)** | **median 7.4 m, 55 percent within 15 m, 28 of 28 anchors, fully GPS free** | real, whole video, mean 76 m because a few anchors lock the wrong tile, the median is robust |
| SIFT baseline | frame 6510 at 90.7 m, about 136 inliers, degenerate, confident yet wrong on repetitive texture | real |
| LoFTR | 1.76 to 1.80 m where it locks, about 15 percent blind coverage, oblique frames fail | real |
| RoMA precision, given a roughly correct tile | 10 of 10 frames, median 1.5 m, best 0.7 m, worst 2.3 m, 45 to 310 times the LoFTR inliers | real GPU bench, precision given a good prior, not a blind search |
| Visual odometry trajectory | 100 percent coverage, 686 of 686 frames, zero chain breaks. Drift dominated: raw trajectory error 165 m, shape aligned RMSE 137 m, length ratio 3.1 (over scaled). A coverage layer, not an accuracy layer, absolute accuracy comes from RoMA | real, GPS free |
| Multi source selection (PNOA and Esri) | 100 percent coverage on feature rich frames, one frame from 19 m to 5 m | real, per frame best source |
| Cross dataset (UAV VisLoc) | region 03 frame at 11.3 m with LoFTR, single frame proof of concept | real, one external dataset frame |
| Partner benchmark | about 10 m across videos described as amazing, we reach about 1.8 m best on matchable frames | context |
| Engineering | 156 offline tests, CI green, Python 3.11 and 3.12 | — |

Numbers vary by about plus or minus 10 m run to run because the matchers are stochastic. This is disclosed in section 9 of the report.

## 3. Methods in the system
Telemetry free framework, pluggable matchers SIFT, LoFTR, eLoFTR, RoMA, pluggable imagery providers PNOA, Esri, Google, GEE, grid search with confidence gate, relative margin lock gate, oblique tilt pose correction, coarse to fine refinement, matcher auto selection, multi source imagery selection, visual odometry, recursive fusion (Kalman and RTS), manual anchoring, trajectory shape metric, GeoJSON and KML export, multi dataset adapters.

## 4. Honest framing
- Fully automated, GPS free accuracy is tens of metres on this low altitude grassy oblique flight, with the blind whole video RoMA pipeline reaching a median of 7.4 m.
- Few metre accuracy is reached where the scene has texture, RoMA reaches about 1.5 m given a roughly correct tile, and LoFTR locks at 1.76 to 1.80 m.
- The edge over a plain pipeline is the automatic selection of both the matcher and the imagery source per frame, plus visual odometry to cover frames the map cannot match directly.
- Keep two RoMA numbers distinct. About 1.5 m is precision given the right tile. The 7.4 m median is the blind whole video result.

## 5. How to run it
```bash
python scripts/run_e2e.py        # fetch video, ingest and shard, GPS, reference, localize
python scripts/run_all.py --providers pnoa,esri --methods sift,loftr,roma --device cuda
```
LoFTR needs torch and kornia, RoMA needs a CUDA GPU (setup in docs/LOCAL_GPU_MATCHANYTHING.md). Both skip gracefully if their dependencies are absent.

## 6. Branches
- `main` is the single working deliverable. All other branches are merged or superseded.
