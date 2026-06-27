# Project status — for the report / poster / presentation team

**Last updated: 2026-06-27** · Single source of truth for numbers + framework state.
Use the numbers in this file. If a figure isn't here, ask before putting it on the poster.

> ⏳ **One result still coming:** the *blind* RoMA numbers from the cloud GPU run
> are being generated now and will be appended here. Everything else below is final.

---

## 1. What the system is (one paragraph for the intro)
A **telemetry-free, modular framework** that localizes a drone from its downward
camera alone, by matching each video frame to georeferenced satellite imagery —
no GPS at runtime (GPS is used only to *score*). Matchers (SIFT / LoFTR / RoMA)
and imagery providers (PNOA / Esri / Google / GEE) are interchangeable behind one
interface, and the framework **auto-selects the best matcher AND the best imagery
source per frame**. One command runs the whole pipeline end-to-end.

## 2. Headline numbers — CONFIRMED REAL (safe to use)

| Capability | Result | How measured |
|---|---|---|
| Per-frame accuracy (LoFTR, feature-rich frames) | **~1.8 m** (best), median 2.6 m | real, Apple-Silicon GPU (MPS) |
| Coverage (LoFTR, blind grid over flight) | ~15 % of frames lock | real, 40-frame scan |
| **Multi-source selection (PNOA+Esri)** | **100 % coverage on feature-rich frames; one frame 19 m → 5 m** | real, per-frame best-source |
| RoMA (precision, tile near truth) | **~1.5 m median** (0.7–2.3), 10/10 frames | real (earlier GPU bench) — *blind cloud number pending* |
| RoMA tile disambiguation | correct tile wins **6/6** (1.8–10.8×) | real |
| Cross-dataset generality (UAV-VisLoc) | **11.3 m** on an external-dataset frame | real |
| Trajectory (VO) | **0.6–2.6 m near absolute fixes**; drifts over long unanchored gaps | real |
| Engineering | **149 offline tests, CI green** (Python 3.11 + 3.12) | — |
| Partner benchmark | "~10 m across videos = amazing" — we beat it on matchable frames | context |

## 3. Methods in the system (list these on the poster)
Telemetry-free framework · pluggable matchers **SIFT / LoFTR / eLoFTR / RoMA** ·
pluggable imagery providers **PNOA / Esri / Google / GEE** · grid search +
confidence gate · **relative-margin lock gate** · **oblique-tilt pose
correction** · **coarse-to-fine refinement** · **matcher auto-selection** ·
**multi-source imagery selection** · visual odometry · **recursive fusion
(Kalman + RTS)** · manual anchoring · trajectory shape metric · GeoJSON/KML
export · multi-dataset adapters (provided video + UAV-VisLoc).

## 4. Honest framing (so the defense holds up)
- **Fully-automated, GPS-free accuracy is tens of metres**, not few-metre — this
  is the real ceiling on this low-altitude/grassy/oblique flight, confirmed
  independently (a peer group's blind automated RoMa was 65 m).
- **Few-metre accuracy needs manual anchoring** (human-marked control points).
  We implemented it, but chose a **fully-automated** system (no human in the loop)
  as the cleaner, telemetry-true deliverable.
- Our edge over a plain pipeline: **auto-selection of matcher AND imagery source,
  plus VO+RoMA fusion** — designed to push the *automated* number below the 65 m
  blind baseline. (Exact blind number: see the pending RoMA cloud run.)
- The RoMA ~1.5 m figure is **precision given roughly the right tile**, not a blind
  end-to-end result — report it as such.

## 5. What CHANGED since the 23 June draft (fix these)
1. **Tests: 79 → 149.**
2. New methods now exist (draft called them future): multi-source selection,
   fusion filter, manual anchoring, margin gate, tilt correction, coarse-to-fine.
3. **Margin gate = done**, not "next step."
4. Generalization now has **real evidence** (UAV-VisLoc 11.3 m) — soften "not demonstrated."
5. Imagery: we use **PNOA (~0.15–0.25 m/px), higher-res than Esri**, and now
   **select per frame** across sources.

## 6. How to run it (the single trigger for the evaluator)
```bash
python scripts/run_all.py --frames-dir <frames> --providers pnoa,esri --device cuda
# -> per-method CSVs, auto_track.csv, track.geojson/.kml, comparison.png,
#    flightpath.png, RESULTS.md   (RoMA runs where its deps exist; skipped gracefully otherwise)
```

## 7. Still PENDING (do not finalize on the poster yet)
- **Blind RoMA numbers on cloud GPU** (running now) — will be appended to §2.
- Final automated trajectory number from RoMA-anchored VO fusion.

## 8. Branches
- `main` — the working framework deliverable.
- `feature/accuracy-loop` — this branch: all the above + tooling. (The written
  **report stays local, never on GitHub**.)
