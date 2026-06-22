# PLAN — match, then exceed, the caspar branch

Updated: 2026-06-10. Roadmap for this branch (`dronomy_loc`) against
`origin/feature/caspar`. Shared facts both branches agree on: true location is
**Asturias, Spain (43.521955, −5.624290)** — the filename dropped the minus;
per-frame GPS ground truth exists in the DJI `djmd` stream via exiftool
(**6853 fixes, GT-only, never a localizer input**).

## 0. New empirical findings (2026-06-10, this branch — from the full djmd decode)

We decoded the complete telemetry (all 6853 samples carry GimbalPitch, drone
attitude, and altitude, not just GPS). Three findings that change the plan:

1. **The oblique-gimbal diagnosis is falsified.** `GimbalPitch` is **−90.0°
   (±0.1°) on every single frame** — the camera was perfectly nadir for the
   whole flight. Caspar's #25/#26 explanation for the ~6 % matchability
   ("planar homography can't align a tilted view") cannot be the cause.
   The real limiter must be **appearance**: low-texture vegetation,
   cross-domain gap (season/lighting vs the orthophoto), and motion blur.
   §3(b) below is rewritten accordingly.
2. **Altitude is constant.** `RelativeAltitude` ≈ 50 m for the entire mission
   (median 50.0 m, max 50.1; below that only during takeoff). The ground
   footprint is therefore *nearly constant* → after one locked match
   calibrates m/px, the 4-scale sweep collapses to a single scale
   (±1 neighbour) — **~4× faster grid search**, and VO scale stays stable.
3. **The flight envelope is tiny: 114 m × 209 m**, never more than **109 m**
   from the filename prior. One ~500 m reference tile covers every candidate
   centre — fetch the "world tile" once, crop candidates locally, zero
   per-candidate network calls.

(GT-only caveat: findings 2–3 inform *engineering choices*; the localizer
itself still never reads telemetry. Finding 1 is diagnosis, not input.)

---

## 1. Where each branch stands

| Capability | diego (this branch) | caspar (`feature/caspar`) |
|---|---|---|
| GT extraction (djmd → CSV) | `telemetry.py` (landed) | done (`gps_track.csv`) |
| Frame ingestion | sharded + resumable, blur-filtered (sharpest-in-window) + `frames.csv` manifest | uniform 1 fps sampling, **no blur filter, no manifest** |
| Reference providers | Esri + PNOA (keyless, landed); GEE (auth pending); IGN-FR legacy | Esri (default), PNOA @ 0.15 m/px, Google (key), GEE, Sentinel |
| Localizer | grid-of-centres × multi-scale, ≥20-inlier lock gate (landed) | same design; gate calibrated ≥20 trust / <10 reject |
| Measured accuracy | none yet — this week | SIFT+PNOA **56.8 m** mean (3-frame); LoFTR+PNOA **70.0 m** mean, best **1.73 m** |
| Coverage analysis | telemetry diagnosis (§0): nadir all flight, cause = appearance | matchability scan: **~6 %** matchable (~frames 6400–6600); his oblique explanation falsified by §0 |
| Accuracy log | starts this week | `ACCURACY_LOG.md` + experiment harness + leaderboard |
| VO / temporal | planned — lever (a) below | declared "not viable" (multi-fix fusion; see §3a for why that verdict is narrower than it reads) |

He is ahead on measurement and calibration; we are ahead on ingestion quality,
frame selection, and now **diagnosis** (§0: the 94 % failure is appearance,
not geometry — his branch doesn't know this yet; share it).

---

## 2. Match phase (this week)

Goal: reproduce his numbers on identical inputs so every later delta is real.

- **Validation harness vs the GPS track.** Score estimates by haversine error
  per frame. Reuse his controlled bench *exactly* so numbers are directly
  comparable: frames **342 / 3083 / 6510**, prior (43.521955, −5.62429),
  grid ±60 m @ 60 m, scales 50/80/110/140, SIFT; and his full-run grid
  (±120 m @ 60 m, scales 60/90/140, 640 px tiles) for LoFTR rows.
- **Replicate the 35-frame matchability protocol**: every 200 frames,
  GT-centred PNOA tile at spans 60/90/140 m, record peak inliers,
  matchable := ≥20. Confirms our pipeline sees the same ~6 % he does.
- **Start our own `ACCURACY_LOG.md`** in his row format (date, change,
  per-frame err, mean, Δ vs prev, commit). One change per row, measured.
- **LoFTR on the matchable segment** (frames ~6300–6700, denser than his
  200-frame stride): map the segment's true extent and best-case precision —
  this segment is the anchor for everything in §3.
- Exit criterion: our SIFT and LoFTR means within noise of 56.8 m / 70.0 m on
  the same frames. If they aren't, the pipelines differ — fix before exceeding.
- **MET 2026-06-10** (see ACCURACY_LOG.md): SIFT 55.3 m (his 56.8), LoFTR
  67.7 m (his 70.0), frame 6510 locked at 1.76 m (his 1.73). The
  telemetry-informed single-scale grid gives the identical lock outcome
  1.8× faster.

---

## 3. Exceed phase — three levers, honestly assessed

### (a) VO dead-reckoning anchored on the sub-2 m segment
- His "trajectory fusion not viable" verdict was about **multi-fix fusion**:
  triangulating among several independent absolute fixes spread along the path.
  That verdict is correct — there is one matchable segment, nothing to fuse.
  It does **not** rule out **odometry chaining**, which needs only *one* anchor.
- Method: chain frame-to-frame relative motion (homography decomposition over
  the near-planar ground, or essential matrix from consecutive frames) outward
  — forward and backward — from the absolute fixes in the ~6400–6600 segment.
  Metric scale comes from the locked match's m/px at the anchor.
- Honest costs: monocular planar VO drifts (expect a few % of distance
  travelled); chaining backward over ~6000 frames will degrade badly far from
  the anchor. Any new absolute lock resets drift, so lever (b) compounds here.
- Deliverable is an **error-vs-distance-from-anchor curve**, not a uniform
  accuracy claim. Even 50–100 m error over the rest of the flight beats
  "no estimate" for 94 % of frames — this directly attacks the coverage
  ceiling he declared a dead end.

### (b) Appearance-gap handling (the actual root cause of the 6 % ceiling)
- Old diagnosis (his #25/#26): oblique gimbal → homography can't fit.
  **Falsified by §0.1** — the camera was nadir all flight. The ceiling is an
  *appearance* problem: vegetation-heavy low-texture ground, season/lighting
  gap vs the orthophoto date, and motion blur. Different problem, different
  fixes:
- **Cross-modal matchers**: MatchAnything — **now wired as a first-class
  matcher** (`matching/matchanything.py`, `get_matcher("matchanything")`),
  so `--method matchanything` works in scripts 07/08/09 unchanged. Built for
  exactly this domain gap. Real weights run in the dedicated env (Caspar's
  `docker/Dockerfile.matchanything`, zju3dv fork). **Win condition:** the
  matchable fraction rises above 6 % on the 35-stop protocol — measure it.
  Also worth a look: RoMa/DKM dense matchers (imcui exposes them too).
- **Reference-date selection**: PNOA WMS exposes multiple coverages/years;
  pick the date closest in season to the flight. Cheap to test — one config
  change per date, scored on the same bench.
- **Preprocessing**: CLAHE / gradient-domain normalization on both sides
  before matching; shadows and exposure are a big part of the gap. Hours of
  work, measurable on the same bench.
- Honest ceiling: nothing invents texture where there is none (uniform grass
  is uniform in both images). Expect partial coverage recovery; the floor for
  everything else is lever (a).

### (c) Blur- and coverage-aware frame selection (cheap, do first)
- Our ingestion manifest already carries per-frame blur scores; he samples
  uniformly with no sharpness signal at all.
- Join the manifest with the matchability scan: automatically pick the
  sharpest frames inside and near the matchable window for anchoring, and the
  sharpest per window everywhere for VO (sharper frames → more stable
  frame-to-frame features → less drift in (a)).
- Days of work, compounds with both other levers. No purity questions.

Order of attack: (c) → (b)-preprocessing+date (cheap, same-day measurable) →
(a) → (b)-cross-modal. The §0 findings make (a) stronger (constant altitude =
stable VO scale) and (b) cheaper (no rectification machinery needed).

---

## 4. Evaluation protocol

| Metric | Definition |
|---|---|
| Position error | haversine (m) vs GT, per frame |
| Lock precision / recall | gate = ≥20 inliers; "good" := error ≤ 5 m (his definition) |
| Coverage | % of frames with a trusted estimate (locked, or VO within stated drift bound) |
| Runtime | s/frame on CPU (LoFTR ≈ 8 s/tile — report it, don't hide it) |

Success bars, in order:
1. Beat **56.8 m** SIFT mean and **70.0 m** LoFTR mean on the identical
   3-frame bench (same frames, prior, grid, scales, source).
2. Beat his 10-frame held-out aggregate (mean 86.7 m, 2/10 within 5 m).
3. Once VO lands: **full-trajectory RMSE against all 6853 GT fixes** — a
   number his branch cannot produce at 6 % coverage. This is the headline.

All numbers land in our `ACCURACY_LOG.md`, one measured row per change.

---

## 5. Convergence with the caspar branch

Adopt (proven on his branch, measured):
- **PNOA as primary source** with the 0.15 m/px source-aware clamp (−44 % on
  the bench) and Esri as fallback.
- **Inlier-count lock gate** ≥20 trust / <10 reject; anisotropy demoted to a
  degenerate-transform guard only — his held-out run falsified it as a
  correctness signal, we don't relearn that lesson.
- His GT coordinates for frames 342/3083/6510 and the matchability scan
  protocol as the common yardstick.

Keep (ours):
- Blur-filtered sharded ingestion + manifest (he has no equivalent; it feeds
  lever (c) directly).
- Package layout (`dronomy_loc` vs his `dronomy`) — **decision deferred**;
  merging layouts mid-campaign buys nothing and risks both pipelines.

Share:
- The `ACCURACY_LOG.md` row format, so the two logs concatenate cleanly for
  the joint report.
- Bench definitions (3-frame + 35-stop) frozen as-is; any change forks the
  comparability and must be a new named bench.

---

## 6. Open decisions for the professor — ANSWERED (Adrian, 2026-06-10 email)

1. **Telemetry purity → settled.** "The only live input will be the image
   data." The SRT/telemetry may be used to BUILD (dev-time diagnosis, GT)
   but will not exist at runtime — exactly our design (self-calibrating
   scale, GT-only track). No change needed.
2. **Required output → position only; heading is a plus.** We already emit
   yaw — the plus is secured; quantify yaw error to claim it.
3. **THE GRADING METRIC → shape precision, not absolute accuracy.**
   Verbatim: "Precision is more important than accuracy: we want a
   trajectory that looks similar in shape and dimensions to the original
   path, even if it is off the ground truth path by a few meters."
   This is what VO dead-reckoning produces by construction — drift shows up
   mostly as a slowly-varying offset/rotation, while local shape is
   preserved. Our evaluation must add the standard metric for exactly this:
   **ATE after rigid SE(2) alignment** (Umeyama, NO scale correction —
   "dimensions" must match unaligned), plus path-length ratio, plus the
   estimated-vs-GT overlay plot on the orthophoto (the artifact he will
   actually look at).
4. **Satellite source → settled.** One map is enough; manual download via
   the Google Earth desktop app is sanctioned. Our automated keyless fetch
   (Esri/PNOA) exceeds the requirement; do ONE compliance run with a
   manually exported Google Earth image for the report.
5. **VO fusion → officially the bonus.** Already working (100 % coverage).
6. **Incoming from Adrian:** camera intrinsics (focal/FOV, sensor size) and
   an official GT file + SRT (email attachments). Cross-check his GT against
   our extracted track (expect ~0 m: same source), and use intrinsics for an
   undistortion pass / metric-scale sanity check when they arrive.

---

## 7. Finishing plan (post-answers)

Ordered; each step lands with a measured row or an artifact.

1. **Shape-precision evaluation** (the graded metric — highest priority):
   `localize/trajectory.py` with `align_se2(est, gt)` (Umeyama w/o scale),
   `ate(est, gt)` raw + aligned, path-length ratio; extend scripts/08 output
   and add `scripts/09_trajectory_report.py` → overlay plot (est vs GT on
   the PNOA world tile) + metrics block. Offline tests with synthetic
   trajectories (known offset/rotation recovered exactly).
2. **Densify anchors** along the flight (stride the LoFTR keyframe search,
   accept locks ≥20 inliers) → flattens the 51+-hop drift tail; re-run 08;
   new ACCURACY_LOG row (aligned-ATE before/after).
3. **Adrian's attachments**: save SRT + official GT under
   `project_instructions/` (gitignored if large), cross-check vs
   `data/gps_track.csv`, note the delta in ACCURACY_LOG.
4. **Heading bonus**: yaw error vs track bearing (GT-only) on locked frames
   + VO frames; one log row.
5. **Google Earth compliance run**: manual desktop export of the area,
   georeference (corner coords from the export/KML), run the bench against
   it, one log row + a paragraph for the report ("brief-compliant source").
6. **Camera intrinsics** (when they arrive): undistort frames in ingest
   (optional flag), re-run bench — expect a small accuracy gain; log row.
7. **Report + presentation**: structure already written in
   `explained-dronomy.md`; lead with the overlay plot and aligned-ATE,
   then per-frame metre-level locks, then the coverage story (6 % → 100 %).
   Align the joint story with caspar's branch (shared bench + log format).
