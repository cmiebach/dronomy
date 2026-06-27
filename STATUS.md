# Project status — for the poster & presentation team

**Last updated: 2026-06-27** · Single source of truth for numbers and structure.
If a number is not in this file, do not put it on the poster — ask first.

This exists so the poster/slides never carry outdated figures. The written
report (local only, not on GitHub) is being updated to match this; the sections
below mirror what the report will say.

---

## 1. Headline numbers — CONFIRMED REAL (safe to use)

| Metric | Value | How measured |
|---|---|---|
| Per-frame accuracy, LoFTR | **~1.8 m** on matchable (near-nadir) frames | real run, frame 6510 = 1.80 m (Apple-Silicon GPU/MPS) |
| Per-frame accuracy, RoMA | **~1.5 m median** (best 0.7, worst 2.3) | real, 10 random frames (Docker GPU) |
| Coverage, RoMA | **~100 %** of a random frame sample | real, 10/10 frames matched (Docker GPU) |
| Coverage, LoFTR | low — only near-nadir frames (~6 %, refresh in progress) | older 35-frame scan; an updated GPU scan is running |
| Tile disambiguation, RoMA | correct tile wins **6/6** (1.8–10.8× margin) | real (Docker GPU) |
| Trajectory precision (shape-aligned RMSE) | **~12 m** (LoFTR anchors) to ~24 m (SIFT), **100 % coverage** | real, visual-odometry track |
| Cross-dataset generality (NEW) | **11.3 m** on a UAV-VisLoc (region 03) frame, LoFTR | real, external dataset — first generality evidence |
| Partner's benchmark | "~10 m across a range of videos = amazing" — we beat it on this clip | context |
| Offline unit tests | **128** on `main` (deliverable) | CI green, Python 3.11 + 3.12 |

**One-line framing (safe):** a *telemetry-free, modular framework* that matches
drone frames to satellite imagery; RoMA lifts coverage from ~6 % to ~100 % at
~1.5 m; a VO + fusion layer gives a 100 %-coverage, shape-faithful trajectory.

## 2. Methods now in the system (use this list on the poster)

On `main` (the graded deliverable): telemetry-free framework (pluggable
matchers SIFT/LoFTR/eLoFTR/RoMA + imagery providers), grid search + confidence
gate, pose recovery, **visual odometry**, **recursive fusion filter (Kalman +
RTS)**, **manual anchoring**, trajectory shape metric, multi-dataset adapters.

Implemented + tested, **real-validation in progress** (do NOT quote final
numbers yet — see §4): **relative-margin lock gate**, **oblique-tilt pose
correction**, **coarse-to-fine refinement**.

## 3. What CHANGED vs the old (23 June) report draft — fix these on the poster

1. **Unit tests: 79 → 128** (deliverable). The old "79" is outdated.
2. **New methods exist** that the old draft called "future"/"next step":
   fusion filter, manual anchoring, margin gate, tilt correction, coarse-to-fine.
   Add them to the methods list.
3. **Margin gate is DONE**, not "the immediate next step."
4. **Cross-dataset generalization** — old draft said "not yet demonstrated."
   We now have a real UAV-VisLoc data point (11.3 m). Soften that caveat.
5. **LoFTR coverage %** may change slightly from ~6 % once the GPU scan finishes.
6. **RoMA numbers are unchanged** (~1.5 m, ~100 %) — already real, safe to keep.
7. **Report structure is stable** (Intro → Related work → Problem → Data →
   Methodology → Alternatives → Engineering → Experiments → Discussion →
   Ethics → Conclusion). The poster can mirror this section order; only the
   *method list* and *numbers* above need updating.

## 4. Still PENDING — do not put these on the poster as final

- Real-data numbers for the fusion filter, tilt correction, and coarse-to-fine
  (currently validated on simulation/analytic geometry; being re-run on the GPU).
- An updated LoFTR coverage % from the in-progress scan.
- Fresh RoMA numbers are NOT pending — the existing RoMA figures are real.

## 5. Git / branch state (so you know what's where)

- **`main` (GitHub):** the complete working framework + fusion + manual
  anchoring. 128 tests. This is the deliverable.
- **`feature/accuracy-loop` (local, not yet pushed):** the three enhancements
  in §2 (margin gate, tilt correction, coarse-to-fine). 144 tests.
- The written **report stays local** (never pushed to GitHub).
