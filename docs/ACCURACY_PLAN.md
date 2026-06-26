# Post-meeting plan — Adrian sync 2026-06-23

Turning Adrian's guidance into a prioritized implementation plan. Source: the
2026-06-23 meeting notes.

## The reframe that drives everything

Adrian was explicit: **the grade is a generic, robust *framework* across varied
terrain + code quality — NOT per-frame accuracy on the one provided video.** Our
1.75 m (LoFTR, provided video) already beats his bar ("10 m across a range of
videos would be amazing"). So the work now is **generalization, RoMA, and
multi-dataset validation**, not squeezing this single clip further.

| Adrian's priority | What it means for us |
|---|---|
| Generic framework > single-video accuracy | Multi-dataset harness is the headline deliverable |
| Deep matching > hand-tuned classical | RoMA/LoFTR are the path; SIFT is a baseline only |
| RoMA = current SOTA | Make RoMA the primary matcher in `main` |
| Satellite map-matching first, VO second | Keep map-matching central; VO/hybrid is enhancement |
| Most-recent imagery mandatory | Verify provider currency; drop seasonal matching |
| Heading low priority | Use only to constrain yaw jumps |

## Workstreams (prioritized)

### W1 — RoMA as the primary matcher in `main`  [P0, biggest lever]
RoMA is SOTA, handles tilt/cross-modal, and generalizes without per-scene tuning
(Adrian). We've benched it: **10/10 random frames matchable, ~1.5 m median**, vs
LoFTR's ~6% coverage on oblique frames.
- Merge/rebase the MatchAnything branch into `main` (action item: Diego; I can do/support).
- **Add a relative-margin lock gate**: a dense matcher scores 80-400 inliers even on
  *wrong* tiles, so the absolute `>=20` gate is invalid for RoMA. Gate on
  best-vs-2nd-best (or best/median) inlier ratio.
- Wire `--method matchanything --ma-model roma` through `search` / `07_validate` /
  `08_vo_trajectory`.
- Ship a runnable path: the `docker/Dockerfile.matchanything` image; document a
  GPU/native host (RoMA is ~2.7 min/frame under Mac emulation — too slow for grids).

### W2 — Multi-dataset generality harness  [P0, the grading criterion]
- A dataset abstraction = (frame source, reference-imagery fetch, GT track) adapters.
- Add **SATLOC** and **UAV VisLoc** adapters alongside the provided video (target 2-3).
- Run the *same* pipeline unchanged across all → per-dataset error table.
- Flag datasets lacking GT as qualitative-only (Diego's concern).

### W3 — Benchmarking + auto-selection layer  [P1]
Adrian: run multiple algorithms, let the framework pick the best per context.
- Harness runs SIFT / LoFTR / RoMA across datasets → a leaderboard.
- Per-frame/per-context selection by an output metric (inlier margin / lock
  confidence), so the framework is plug-and-play.

### W4 — Most-recent satellite imagery  [P1]
- Verify Esri / Google providers return *current* imagery; document map-currency.
- **Drop seasonal/temporal matching** — Adrian did not endorse it (recency wins).

### W5 — Camera intrinsics reconciliation  [P1, quick win, helps scale]
- Adrian's reference still gives **focal length ~= 3713 px** + principal point
  (DJI XMP/DewarpData). Our `config/camera_mavic3e.yaml` *derived* fx ~= 2664 px
  at 3840 px video. These are consistent IF 3713 is at the 5280 px photo width
  (3713 x 3840/5280 ~= 2700 px) — verify, then pin the exact XMP value (scaled to
  video resolution) for scale recovery. Distortion is low-priority (Adrian).

### W6 — Tilt / terrain robustness  [P2]
- RoMA already mitigates this (LoFTR misreads terrain slope as camera pitch — Adrian).
- Avoid nadir assumptions in code; a 2-3 deg pitch is fine with a wide FOV. Defer
  heavy 3D/DEM correction (out of scope for 6 days).

### W7 — Hybrid (map + VO) and heading constraint  [P2]
- We have both pieces: VO (100% coverage) + satellite absolute anchors. Hybrid =
  VO continuous + map corrections when available. Use heading only to reject
  implausible yaw jumps. VO stays secondary per Adrian.
- **DONE — recursive fusion filter** (`localize/fusion.py`, `tests/test_fusion.py`):
  constant-velocity Kalman filter + RTS smoother over `[east, north, v_e, v_n]` in a
  local metre plane. Fuses intermittent absolute fixes (and optional VO velocity),
  **bridges unlocked gaps**, and **chi-square-gates outlier locks** (the lock-to-the-
  wrong-building failures). API: `fuse_track(steps)` (generic) and
  `fuse_frame_scores(rows)` (takes `validate.FrameScore`, returns a fused position
  for *every* frame). Telemetry-free: only the system's own visual fixes enter.
  Measured on the real flight geometry with a dense fix stream (sim: 70% lock, 5 m
  noise, 10% outliers): median **6.0 -> 1.6 m**, worst **386 -> 5.8 m**, 100 outliers
  gated, full per-frame coverage. Best paired with W1 (RoMA gives the dense fix
  stream the filter needs to shine).

### W8 — Feature-stability evaluation  [P3]
- Prefer rivers/terrain/rocks; treat roads/construction as unreliable. Mostly an
  eval/reporting lens for the non-urban target environment.

## Already done (maps to Adrian's asks)
Sharded + integrity-verified ingestion; DJI-telemetry GT; coordinate fix
(Spain); multi-source imagery (Esri/PNOA/Google/GEE); SIFT/LoFTR/RoMA implemented
(RoMA benched 10/10); camera metadata to JSON; VO at 100% coverage; trajectory
**shape** metric (Umeyama-aligned RMSE = Adrian's "looks like the path" criterion).

## Recommended immediate three (accuracy + grade impact)
1. **W1** — RoMA into `main` + margin gate (the matcher Adrian named as SOTA).
2. **W2** — multi-dataset harness with one extra dataset (SATLOC) wired in.
3. **W5** — intrinsics reconciliation against the 3713 px reference (fast).
