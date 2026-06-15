# Accuracy log — diego branch

Chronological, measured record: one change per row, scored against the GPS
ground truth (`data/gps_track.csv`, extracted from the DJI `djmd` stream —
**scoring only, never a localizer input**). Same row format as the caspar
branch's `ACCURACY_LOG.md` so the two logs concatenate for the joint report.

## Controlled bench (frozen — identical to caspar's, numbers directly comparable)

- **Frames:** 342, 3083, 6510 (6510 = the known degenerate-SIFT case).
- **Coarse prior (telemetry-free):** lat 43.521955, lon −5.624290 (filename).
- **SIFT grid:** ±60 m @ 60 m (9 centres), scales 50/80/110/140 m, 640 px tiles.
- **LoFTR grid (his "full-run" grid):** ±120 m @ 60 m (25 centres), scales
  60/90/140 m, 640 px tiles.
- **Source:** PNOA, cropped locally from ONE 600 m @ 4096 px world tile
  (0.146 m/px) — telemetry showed the whole flight sits within ~109 m of the
  prior, so one tile covers every candidate (zero per-candidate network calls).
- **Error:** haversine (m) vs the GPS fix of the same frame. Lock gate ≥20
  inliers (his calibration: 4–9 = noise floor).
- Hardware: Windows laptop, CPU only (torch 2.12 cpu, kornia 0.8.3 LoFTR
  "outdoor").

GT for the bench frames (from our extraction; matches caspar's to 0.00 m):
| frame | gt_lat | gt_lon |
|---|---|---|
| 342 | 43.52195843 | −5.62429169 |
| 3083 | 43.52128088 | −5.62378552 |
| 6510 | 43.52217906 | −5.62556347 |

## Change log

| # | Date | Change | 342 | 3083 | 6510 | Mean | Reference | Commit |
|---|---|---|---|---|---|---|---|---|
| 0 | 2026-06-10 | **SIFT baseline** (his SIFT grid) | 20.0 (39 inl) | no pose | 90.7 (133 inl ⚠) | **55.3 m** | his 56.8 m — reproduced | ca64b85 |
| 1 | 2026-06-10 | **LoFTR baseline** (his LoFTR grid) | 106.7 (15 inl, unlocked) | 94.7 (15 inl, unlocked) | **1.76 (117 inl, LOCKED)** | **67.7 m** | his 70.0 m — reproduced | (this) |
| 2 | 2026-06-10 | **Single-scale grid** (70 m only, step 40; telemetry showed alt ≈ 50 m const → footprint ≈ 71 m measured) | no pose | no pose | **1.75 (35 inl, LOCKED)** | locked-only: 1.75 m | **1.8× faster** (153 s vs 278 s/frame), identical lock outcome | (this) |

### Reading the rows honestly

- **Row 0** confirms the whole pipeline (ingest → world tile → grid search →
  scoring) reproduces his SIFT numbers within noise, including the documented
  6510 failure signature: 133 inliers but 90.7 m off — high-count SIFT locks
  on repetitive structure are NOT trustworthy, which is why the gate is on
  *deep*-matcher inliers.
- **Row 1** reproduces his headline: LoFTR + PNOA nails the matchable frame at
  1.76 m and correctly leaves 342/3083 below the lock gate (15 inliers = junk
  poses; the gate rejects them). Mean-over-3 is dominated by unlocked junk —
  lock precision is the metric that matters.
- **Row 2** is our telemetry-informed speedup: constant altitude ⇒ constant
  footprint ⇒ one scale suffices. Same trusted result (6510 locks at 1.75 m),
  1.8× faster, and the untrustable frames now yield *no pose at all* instead
  of sub-gate noise — precision over recall, for free.

## VO dead-reckoning (full-trajectory; the coverage lever)

Setup: anchors from LoFTR grid search (telemetry-free, lock gate ≥20) at
keyframes 6400/6500/6600; sweep every 10th frame (≈3 fps, 686 frames); links =
SIFT homographies between consecutive swept frames (min 30 inliers); chain =
compose links to the nearest anchor, georeference through the anchor's tile
registration. Scored per swept frame vs GT. Run 2026-06-10
(`scripts/08_vo_trajectory.py --provider pnoa --method loftr`).

**Anchors (all telemetry-free locks):**
| keyframe | err vs GT | inliers | search time |
|---|---|---|---|
| 6400 | 2.54 m | 33 | 154 s |
| 6500 | 1.70 m | 38 | 160 s |
| 6600 | 1.59 m | 57 | 158 s |

**Chain:** 686 swept frames, 685/685 links held (0 breaks), 137 s for the
whole sweep.

**Full-trajectory result (all 686 frames scored vs GT):**

| metric | value |
|---|---|
| **Coverage** | **686/686 = 100 %** (vs ~6 % per-frame-matching ceiling) |
| Median error | 12.3 m |
| Mean / RMSE | 26.1 m / 35.7 m |
| Worst frame | 70.2 m |

**Drift curve (error vs hops from the nearest anchor):**
| hops | n | median | worst |
|---|---|---|---|
| 0–10 | 41 | 1.6 m | 4.2 m |
| 11–50 | 55 | 4.2 m | 4.4 m |
| 51+ | 590 | 30.3 m | 70.2 m |

Reading it:
- Near the anchors the chain is **metre-level** (1.6 m median) — VO barely
  degrades the anchor accuracy for ~±50 frames (~17 s of flight).
- The 51+ bucket is dominated by chaining **backwards through the entire
  flight** from anchors that all sit in the final segment — up to ~640 hops.
  Even then the worst frame (70.2 m) is comparable to the SIFT *mean* on
  matchable frames.
- The lever to flatten the tail is **more anchors spread along the flight**
  (each lock resets drift); coverage itself is already solved by the chain.
- Caveat: drift here is benign partly because the flight is slow (~1 m/s)
  and the scene static; the error-vs-hops curve, not the headline mean, is
  the honest deliverable. CSV: `data/outputs/vo_trajectory.csv`.

## Shape-precision metric (Adrian's graded criterion)

`scripts/09_trajectory_report.py` scores the same VO run the way Adrian asked
("similar in shape and dimensions, even if off by a few meters") — rigid SE(2)
alignment (rotation+translation, NO scale), then ATE. Figure:
`docs/figures/trajectory_report.png`.

| metric | value | meaning |
|---|---|---|
| ATE raw | 35.7 m | before alignment (penalizes constant offset) |
| **ATE SE(2)-aligned** | **27.6 m** | the graded shape metric (686 frames) |
| Path-length ratio | **0.91** | est path is 444 m vs GT 488 m — dimensions ~right |
| Heading offset | −6.2° | constant bias the rigid align removes |

Honest read: the loop shape is clearly recovered (see the figure — both tracks
trace the same circuit), but pure VO drifts because **all three anchors sit in
the final segment** (frames 6400–6600), so most frames chain 100s of hops back.
The aligned 27.6 m is dominated by that one-sided anchoring. **Lever:** the
appearance-gap work (PLAN §3b) is what unlocks anchors earlier in the flight;
each new anchor cuts the chain length and tightens the shape. This row is the
"before" — re-measured after each anchor added.
