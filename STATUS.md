# Project status — for the report / poster / presentation team

**Last updated: 2026-06-28** · Single source of truth for numbers + framework state.
Use the numbers in this file. If a figure isn't here, ask before putting it on the poster.

> **RoMA note:** RoMA runs **natively on the Apple-Silicon GPU (MPS)** — no Docker
> needed. The **full-video blind run completed**: RoMA selected on all 28/28 anchors,
> chained over the whole video → **median 7.4 m, 55 % within 15 m, GPS-free** (mean
> 76 m — a few anchors lock the wrong tile confidently; the median is robust). Two
> distinct RoMA numbers, don't conflate them: **~1.5 m is *precision* given the right
> tile**; **~7.4 m (median) is the *blind* whole-video result.**
> RoMA also now runs **locally on a CUDA GPU** (RTX 3080 Ti, no Docker/cloud; setup:
> `docs/LOCAL_GPU_MATCHANYTHING.md`). In the **cascade** it refines LoFTR's locks to
> **median ~2.0 m (4/4 < 5 m**, one frame LoFTR had at 16.5 m → 1.3 m) and recovers
> frames LoFTR missed at sub-metre, lifting blind recall@5m **0.15 → 0.35**.

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
| **Full-video pipeline (RoMA per-frame, blind)** | **median 7.4 m · 55 % within 15 m · 28/28 anchors** | real, WHOLE video, GPS-free (mean 76 m — a few wrong-tile anchors; median is robust) |
| Per-frame accuracy (LoFTR, feature-rich frames) | **~1.8 m** (best), median 2.6 m | real, Apple-Silicon GPU (MPS) |
| Coverage (LoFTR, blind grid over flight) | ~15 % of frames lock | real, 40-frame scan |
| **Multi-source selection (PNOA+Esri)** | **100 % coverage on feature-rich frames; one frame 19 m → 5 m** | real, per-frame best-source |
| RoMA per-frame coverage / precision | **100 % matched (10/10); ~1.5 m** given the right tile | real GPU bench — *that 1.5 m is precision given a good prior, NOT blind* |
| RoMA cascade (**local CUDA, RTX 3080 Ti**) | refine **4/4 < 5 m, median ~2.0 m**; recovery lifts blind recall@5m **0.15 → 0.35** | real, local GPU end-to-end cascade |
| RoMA tile disambiguation | correct tile wins **6/6** (1.8–10.8×) | real |
| VO flight path (shape) | **137 m shape-aligned RMSE, path-length ratio 3.11**, 686 frames, continuous | real, GPS-free — VO over-scales ~3×; absolute re-anchoring bounds error (data: `data/outputs/vo_trajectory.csv`) |
| Cross-dataset generality (UAV-VisLoc) | **11.3 m** on an external-dataset frame | real |
| Trajectory (VO) | **0.6–2.6 m near absolute fixes**; drifts over long unanchored gaps | real |
| Engineering | **156 offline tests, CI green** (Python 3.11 + 3.12) | — |
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
  is the real ceiling on this low-altitude/grassy/oblique flight.
- **Few-metre accuracy needs manual anchoring** (human-marked control points).
  We implemented it, but chose a **fully-automated** system (no human in the loop)
  as the cleaner, telemetry-true deliverable.
- Our edge over a plain pipeline: **auto-selection of matcher AND imagery source**.
  The blind whole-video RoMA pipeline reaches **median 7.4 m** — strong for a
  fully-automated, GPS-free result (a few wrong-tile anchors pull the mean up).
- Keep the two RoMA numbers distinct: **~1.5 m = precision** given the right tile;
  **~7.4 m median = blind** whole-video localization. Don't present 1.5 m as blind.

## 5. What CHANGED since the 23 June draft (fix these)
1. **Tests: 79 → 156.**
2. New methods now exist (draft called them future): multi-source selection,
   fusion filter, manual anchoring, margin gate, tilt correction, coarse-to-fine.
3. **Margin gate = done**, not "next step."
4. **Generalization = one UAV-VisLoc frame (11.3 m), a single-frame proof of concept.** Do NOT claim broad cross-terrain generalization; keep "not yet demonstrated."
5. Imagery: we use **PNOA (~0.15–0.25 m/px), higher-res than Esri**, and now
   **select per frame** across sources.

## 6. How to run it (the single trigger for the evaluator)
```bash
python scripts/run_all.py --frames-dir <frames> --providers pnoa,esri --device cuda
# -> per-method CSVs, auto_track.csv, track.geojson/.kml, comparison.png,
#    flightpath.png, RESULTS.md   (RoMA runs where its deps exist; skipped gracefully otherwise)
```

## 7. Status of optional extras
- **Blind full-video RoMA: DONE** natively on the Mac GPU (MPS) — median 7.4 m
  (figure: `docs/figures/full_flight_path_roma.png`). The figure has a few spike
  artifacts from wrong-tile anchors; a margin-gated re-run would clean it.
- Cleaner RoMA figure (margin gate to drop wrong-tile anchors) = nice-to-have.

## 8. Branches
- `main` — the working framework deliverable.
- `feature/accuracy-loop` — this branch: all the above + tooling.
