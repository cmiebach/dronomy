# Dronomy: Telemetry Free Visual Localization of Drones in GNSS Denied Environments

IE University capstone project, in partnership with Dronomy. Group 3: Aylin Yasgul, Caspar Miebach, Diego Alfaro, Yi Long, Alessandro Cristofolini.

> All headline numbers in this document mirror [`STATUS.md`](STATUS.md), which is the single source of truth. If a figure is not in STATUS.md, do not put it on the poster or in the report.

## Abstract

Most drones rely on the Global Navigation Satellite System for positioning, yet that signal is unavailable, degraded, or jammed in many real settings, from the move between indoor and outdoor flight to dense terrain, urban canyons, and contested airspace. This project presents a telemetry free system that estimates a drone's absolute position, latitude and longitude, and optionally its heading, from a downward looking camera alone, by matching each video frame to a georeferenced satellite map. No GPS is read at runtime. The contribution is not a single tuned solution but a generic, modular framework: matchers and imagery providers are interchangeable behind stable interfaces, and the same pipeline runs unchanged across terrains and datasets. On the provided drone flight over Asturias in Spain, the blind whole video pipeline reaches a median error of 7.4 m fully GPS free, individual feature rich frames localize to about 1.8 m, and the identical code generalizes to an external dataset frame in China. These results sit well inside the partner benchmark of about 10 m across videos.

## 1. The problem

Outdoor autonomy almost always depends on GNSS, which is a single point of failure: the signal is weak near structures and terrain, can be degraded by multipath, and can be deliberately jammed or spoofed. Indoor autonomy avoids GNSS but usually alters the environment with markers or beacons. The partner, Dronomy, builds GPS denied autonomy that does not alter the flight area, and this project extends that idea to the outdoor setting.

The core observation is simple. A downward looking camera over open terrain sees the same ground that a satellite sees from above. If a drone frame can be matched reliably to a georeferenced satellite map, the drone's absolute position can be read off directly, with no GNSS, no markers, and no change to the environment. The difficulty is that the two images come from different sensors, times, seasons, resolutions, and viewing angles. Bridging that appearance gap robustly is the heart of the work.

## 2. Approach

The system is a configuration driven package, `dronomy_loc`, built so that swapping the matcher or the imagery provider requires no change elsewhere. That interchangeability is what makes it a framework rather than a one off, and it is the lever for generality. The pipeline is:

ingest video, extract and verify frames, fetch a georeferenced reference tile, match frame to map, estimate a homography, recover pose as latitude, longitude, heading, and scale, search a grid of candidates with a confidence gate, score against the GPS track, and optionally fuse with visual odometry.

Three design choices carry most of the weight:

* **Pluggable matchers.** SIFT is a fast classical baseline. LoFTR is a detector free transformer that is strong on low texture. RoMA, a dense matcher trained for cross modality, is the lever that lifts coverage across the appearance gap. The framework selects the best matcher per frame.
* **Pluggable imagery.** Esri World Imagery is keyless and global. Spanish IGN PNOA orthophoto is the highest resolution source over the true flight area at 0.15 to 0.25 m per pixel. Google Earth Engine provides Sentinel 2. The framework selects the best source per frame.
* **Trust before coverage.** A fix is accepted only with at least twenty deep matcher inliers, filtered by a relative margin gate so a dense matcher cannot lock confidently onto the wrong tile. Visual odometry then chains from confirmed matches to cover frames the map cannot match directly.

The drone's own GPS is used only to score error, never as an input. The system stays telemetry free by design.

## 3. Results

The canonical table lives in [`STATUS.md`](STATUS.md). In summary:

* **Blind whole video pipeline, the headline:** median 7.4 m, 55 percent of frames within 15 m, RoMA selected on all 28 of 28 anchors, fully GPS free.
* **Single frame accuracy:** LoFTR locks at 1.76 to 1.80 m where the scene has texture, RoMA reaches about 1.5 m median precision given a roughly correct tile, over a 0.7 to 2.3 m range across 10 of 10 frames.
* **Coverage:** about 15 percent of frames lock with a single blind matcher, and multi source selection plus visual odometry raises coverage to 100 percent on feature rich frames.
* **Generality:** the identical code localizes an external dataset frame in China to 11.3 m with LoFTR, a single config line apart from the Spain flight.
* **Engineering:** 156 offline tests, continuous integration green on Python 3.11 and 3.12.

The honest framing is that fully automated, GPS free accuracy is in the tens of metres on this low altitude, grassy, oblique flight, and few metre accuracy is reached where the scene has texture. Because deep matchers and the geometric estimator are stochastic, individual figures vary by about plus or minus 10 m on re execution.

## 4. How to run it

A fresh clone needs nothing staged. One command downloads the flight video, shards it, extracts the GPS ground truth, fetches reference imagery, and localizes.

```bash
pip install -e ".[dev]"
python scripts/run_e2e.py
```

This writes `data/outputs/run_all/RESULTS.md` with per method tables, the auto selected track, GeoJSON and KML exports, and figures. To run only the localizer once the data is present:

```bash
python scripts/run_all.py --providers pnoa,esri --methods sift,loftr,roma --device cuda
```

LoFTR needs torch and kornia. RoMA needs a CUDA GPU, with setup in [`docs/LOCAL_GPU_MATCHANYTHING.md`](docs/LOCAL_GPU_MATCHANYTHING.md). Both skip gracefully when their dependencies are absent. The full offline test suite runs with `pytest`, with no network, GPU, or video required.

## 5. Repository structure

```
config/config.yaml            central configuration, paths, provider, matcher, search
src/dronomy_loc/
  data/                       video reading, sharded ingestion, DJI GPS via exiftool
  reference/                  Web Mercator geo math, provider interface, Esri, PNOA, GEE
  matching/                   SIFT, LoFTR, and the MatchAnything RoMA backend
  localize/                   homography to pose, grid search, validation, odometry, fusion
  eval/                       field metrics, recall at threshold, best model selection
  export/                     GeoJSON and KML track export
scripts/                      numbered steps plus run_e2e.py and run_all.py entrypoints
tests/                        offline tests, synthetic videos, mocked network
data/                         generated artifacts, git ignored
```

## 6. Documentation

* [`STATUS.md`](STATUS.md): the single source of truth for all numbers and status.
* [`explained-dronomy.md`](explained-dronomy.md): the long form walkthrough of what, why, and how.
* [`STRUCTURE.md`](STRUCTURE.md): the repository and code structure.
* [`docs/LOCAL_GPU_MATCHANYTHING.md`](docs/LOCAL_GPU_MATCHANYTHING.md): running RoMA on a local CUDA GPU.
* [`docker/RUNPOD.md`](docker/RUNPOD.md): running RoMA on a cloud GPU pod.
* [`CONTRIBUTING.md`](CONTRIBUTING.md): conventions and contribution rules.
