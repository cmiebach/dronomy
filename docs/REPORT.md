# Telemetry-free Visual Localization of Drones in GNSS-denied Environments

**IE MBDS Capstone Project · Partner: Dronomy**

**Team (Group 3):** `[TODO: full names + student IDs]`
**Academic tutor:** Hind `[TODO surname]` · **Company mentor:** Dr. Adrián Carrio (Co-founder & CEO, Dronomy)
**Submission:** 29 June 2026

> Formatting note for the team: this is the complete content draft. Convert to the
> required final format — **10–20 pages, 1.0 spacing, 12-pt Times New Roman, PDF**,
> ethics statement on the cover. `[TODO]` marks the few items only the team can
> supply (names, the rendered figures, and the multi-dataset numbers once those runs
> finish). Figures referenced as *Figure N* are produced by `scripts/10_figures.py`
> and `outputs/ma_bench/overlay_crop.jpg`.

### Ethics statement *(place on the cover page)*
This project develops absolute drone localization for **GNSS-denied environments** —
an explicitly dual-use capability spanning search-and-rescue, infrastructure
inspection, precision agriculture, and defense/navigation in jammed or contested
airspace. We commit to responsible use and note: **(1) No personal data** — the
system processes only aerial terrain imagery and public/commercial satellite
basemaps; no identifiable individuals are involved. **(2) Telemetry-free by design** —
the localizer never consumes GPS/telemetry at runtime (GPS is used only offline, as a
scoring reference), so the method neither depends on nor expands surveillance
infrastructure. **(3) Licensed imagery** — satellite/orthophoto sources are used
within their terms (Esri World Imagery; Spanish IGN PNOA open data; Google Earth).
**(4) Dual-use acknowledgment** — we recognise the defense applications discussed with
the partner and frame the work for civilian-protective and humanitarian contexts.

---

## Abstract

Most drones depend on GNSS for positioning, but GNSS is unavailable, degraded, or
jammed in many real settings. We present a **telemetry-free** system that estimates a
drone's **absolute position (latitude/longitude)** — and optionally heading — from a
**bottom-looking camera alone**, by matching each video frame to a **georeferenced
satellite map**. The contribution is a **generic, modular framework** (pluggable
matchers and imagery providers) rather than a solution tuned to one clip. On the
provided DJI Mavic 3 Enterprise flight over Asturias (Spain), a deep matcher (LoFTR)
localizes matchable frames to **~1.8 m**; the state-of-the-art cross-modal matcher
**RoMA** lifts *coverage* from ~6 % to **100 % of a random frame sample at ~1.5 m
median error**; and a **visual-odometry** layer anchored on absolute fixes yields a
**100 %-coverage trajectory**. These results comfortably exceed the partner's
"10 m across videos would be excellent" bar on the provided video; the remaining work
is cross-dataset generalization, which the framework is explicitly built for.

---

## 1. Objective

**Problem.** Commercial drones navigate with GPS/GNSS, but GNSS fails indoors→outdoors,
under dense terrain, and in jammed/contested airspace. Dronomy builds GPS-denied
autonomous flight; this project addresses the **outdoor** case.

**Aim.** Given (a) aerial RGB video from a drone's **bottom-looking (nadir) camera**
and (b) a **georeferenced satellite map** of the area, estimate the drone's **absolute
pose** — latitude/longitude (required), heading (secondary) — for each frame, with **no
GPS or telemetry input at runtime**.

**Success criteria (partner-defined, meeting of 2026-06-23).** The deliverable is a
**generic, robust framework** that works across varied terrain (forest, rivers, rocks,
fields), *not* a single-video solution. **Code quality and generality outweigh
per-frame accuracy.** Reference bar: *"10 m average error across a range of videos
would be amazing."* Position matters more than heading; a **trajectory whose shape and
dimensions match the true path** (precision) is valued above a small absolute offset.

---

## 2. Data sources

**2.1 Drone footage (the only runtime input).** One DJI **Mavic 3 Enterprise (M3E /
WM265E)** clip: 4K **3840×2160**, **29.97 fps**, **6 853 frames**, 228.7 s, H.264;
bottom-looking wide camera; flown over **Asturias, northern Spain** (≈ 43.5220 N,
−5.6243 W). An early sign error in the filename mislabeled the site as France; corrected
to Spain, which determines the correct imagery provider. The partner confirmed that **at
deployment the only available signal is the image stream**.

**2.2 Ground truth — scoring only, never a model input.** The MP4 embeds DJI telemetry
as a protobuf stream (`dvtm_wm265e.proto`). We decode a **per-frame GPS + altitude**
track for all 6 853 frames with `exiftool -ee` (`GPSLatitude/Longitude`,
`AbsoluteAltitude`, `RelativeAltitude`). The decoded track **matches the project's GPS
reference exactly (max difference 0.00 m)** and recovers altitude the original track
lacked: a near-constant cruise of **~49 m above ground level** (take-off 6.6 m → ~50 m).
This is used solely to measure error and to inform scale priors (Section 3).

**2.3 Reference satellite imagery (the map).** A pluggable provider layer, each
returning a georeferenced tile with an exact pixel↔lat/lon mapping:
- **Esri World Imagery** — keyless, global, sub-meter (default).
- **Spanish IGN PNOA orthophoto** — keyless WMS, **~0.15–0.25 m/px** over Asturias (best
  resolution here; used for the accuracy numbers).
- **Google** (Static Maps / Earth / Earth Engine) — partner-suggested; a single manual
  Google-Earth export is acceptable per the brief.
Per the partner, **map currency is mandatory** (deployment is "now"); seasonal/temporal
matching was considered and **dropped** (recency takes priority).

**2.4 Camera intrinsics.** Identified from the container (`encoder = DJI DJIMavic3
Enterprise`) and the partner's reference image: M3E wide camera, **4/3 CMOS
(17.3 × 13.0 mm)**, 24 mm-equivalent, DFOV 84° (photo). **Focal length ≈ 3 713 px** at
the 5 280 px photo width, → **≈ 2 700 px at the 3 840 px video** — giving a ground
footprint of **~70 m** across at 49 m AGL and a GSD of **~1.8 cm/px**. These set the
tile-span prior for the search. `[TODO: pin exact CalibratedFocalLength + principal
point from the reference still's DJI XMP.]`

**2.5 Generalization datasets (in progress).** **SATLOC** and **UAV-VisLoc**
(partner-recommended) for cross-terrain validation. `[TODO: ingest + confirm ground-truth
availability.]`

---

## 3. Methodology

**3.1 Framework.** A config-driven, plug-and-play pipeline implemented as the installable
package `dronomy_loc`:
`ingest video → extract/verify frames → fetch georeferenced reference → match frame ↔
map → homography → pose (lat/lon/yaw/scale) → grid search + lock gate → validate vs GPS →
(optional) visual-odometry fusion`. Sub-packages: `data/`, `reference/`, `matching/`,
`localize/`, `viz/`; driven by `config/config.yaml`; runnable via numbered
`scripts/01..10`. The design goal — *swap the matcher or the imagery provider without
touching the rest* — is what makes the system a framework rather than a one-off.

**3.2 Frame ingestion.** Sharded (~30 s shards), resumable, **SHA-1 integrity-verified**,
with **blur-aware selection** (keep the sharpest frame per window) so out-of-focus frames
do not poison matching.

**3.3 Matching (the core).** A pluggable `Matcher` interface with four backends, chosen to
span the cost/robustness spectrum:
- **SIFT** (classical; Laplacian-of-Gaussian keypoints) — fast, strong for same-modality
  imagery, but appearance- and texture-dependent and prone to confident-but-wrong matches
  across the drone↔satellite domain gap.
- **LoFTR** (transformer, detector-free, semi-dense) — robust on low-texture scenes;
  moderate cost.
- **Efficient LoFTR (eLoFTR)** — a faster LoFTR variant.
- **RoMA / MatchAnything** (dense, DINOv2 backbone) — **state of the art** for
  cross-modality and oblique viewpoints; the lever for the coverage ceiling.
Correspondences feed a homography fit with **MAGSAC++** (robust RANSAC), which gives
stabler inlier counts near the low cross-modal floor.

**3.4 Search and confidence gate.** Because the coarse prior is offset and altitude (hence
scale) is unknown, we run a **grid-of-centres × multi-scale** search around the prior and
keep the candidate with the most inliers. For sparse matchers the **absolute inlier count**
gates a "lock" (~20 = trusted; 4–9 = noise floor). For **dense** matchers (RoMA) absolute
counts are not separable (wrong tiles also score high), so a **relative-margin** gate
(best vs second-best) is required. *(The margin gate is the next integration step.)*

**3.5 Visual odometry (secondary).** Consecutive frames match trivially in the same
modality (SIFT yields 5 000–10 000 inliers/pair). We estimate frame-to-frame motion and
integrate it into a dense trajectory, **anchored on satellite-locked frames** to bound
drift — the brief's bonus "fuse with visual odometry." Heading is used only to reject
implausible yaw jumps.

**3.6 Evaluation metric.** Beyond per-frame haversine error, we report a
**trajectory-shape** metric: align the estimate to the GPS track with a similarity/rigid
transform (Umeyama/Procrustes) and measure the residual RMSE — directly encoding the
partner's "matches the path in shape and dimensions" criterion. Correctness is guarded by
**79 offline unit tests** (geo math, ingestion, telemetry, providers, search, validation,
VO), run automatically on **every pull request** via GitHub Actions (Python 3.11 + 3.12).

---

## 4. Potential solutions (alternatives considered)

The brief requires at least two alternatives with rationale. We evaluated five and chose a
deep-matching, hybrid design.

| Alternative | Idea | Strength | Weakness | Verdict |
|---|---|---|---|---|
| **A. Classical map-matching (SIFT)** | keypoint match frame↔map | fast, no GPU/deps | cross-modal & low-texture failures; **degenerate locks** (frame 6510: 136 inliers but **90.7 m wrong**) | baseline only |
| **B. Deep map-matching (LoFTR)** | transformer, detector-free | locks matchable frames to **~1.8 m** | only ~6 % of (oblique) frames register; tilt-sensitive | strong, low coverage |
| **C. Deep map-matching (RoMA)** ★ | dense, DINOv2, cross-modal | **10/10 random frames matchable, ~1.5 m**; 45–310× more inliers than LoFTR; tilt-robust | heavy (GPU/Docker) | **chosen primary matcher** |
| **D. Visual odometry only** | chain frame-to-frame motion | continuous, no map needed | **drifts** (no absolute reference) | secondary |
| **E. Hybrid (C + D)** ★ | RoMA absolute fixes + VO between | continuous **and** drift-corrected | most engineering | **chosen architecture** |

**Rationale.** The partner identified RoMA as SOTA and prioritised generality over
per-scene tuning; deep matching avoids hand-calibration and so generalises across terrain.
RoMA removes the coverage ceiling (~6 % → ~100 %) while holding ~1–2 m precision, and VO
fills the gaps and smooths the path. SIFT is retained as a fast, dependency-free baseline
for the benchmarking layer; LoFTR is the mid-tier deep matcher when a GPU is unavailable.

---

## 5. Development (code & engineering)

- **Package:** `dronomy_loc` (src-layout, `pip install -e .`); clean abstract interfaces
  (`Matcher`, `ReferenceProvider`) + factories; a typed `GeoImage` carries the
  pixel↔lat/lon contract, removing scattered bbox handling.
- **Reproducibility:** one `config/config.yaml`; numbered `scripts/01..10` as "small
  working pieces" (extract → reference → localize → run-video → ingest → GPS track →
  validate → VO → report → figures); all data git-ignored (no data ever in git).
- **Quality gates:** **79 offline unit tests** (synthetic videos, mocked network — no torch
  required) plus **GitHub Actions CI** running them on **every PR** across Python 3.11/3.12;
  the test check is **required** by the branch ruleset, so no failing change can merge.
- **Robust failure modes:** integrity-verified ingestion; per-tile error tolerance in the
  search; and missing deep-matcher dependencies now **fail loudly** (clear message) instead
  of silently reporting "0 inliers".
- **Deep/RoMA environments:** LoFTR needs torch + kornia (Python ≤ 3.12); RoMA runs via
  `docker/Dockerfile.matchanything` (zju3dv weights), isolated so the core package and the
  test suite never depend on it.
- `[TODO: add repo URL + an accessible notebook walkthrough (the brief asks for a notebook).]`

---

## 6. Results & evaluation

All figures below are on the provided clip with the PNOA reference. *(Cross-dataset numbers
are pending — Section 7.)*

**6.1 Per-frame localization.**

| Method | Matchable frame (6510) | Coverage on random frames | Note |
|---|---|---|---|
| SIFT (classical) | 90.7 m (136 inliers, **degenerate**) | low | confident-but-wrong locks |
| **LoFTR** | **1.76–1.80 m** (165 inliers) | ~6 % (oblique frames fail to lock) | tilt-sensitive |
| **RoMA** | sub-2 m | **10/10 frames; median 1.5 m, best 0.7 m, worst 2.3 m** | 45–310× LoFTR inliers |

The decisive result: on **10 random frames sampled across the whole flight** (including the
oblique, forward-flight frames LoFTR cannot register), **RoMA matched all 10** and localized
them to **1.5 m median**. *Figure 1* overlays RoMA estimates (red) on ground truth (green) on
the orthophoto — the points are near-coincident at ~1–2 m.

**6.2 Trajectory (visual odometry).** Anchored on satellite-locked frames, VO produces a
**100 %-coverage** trajectory (686/686 sampled frames, 0 chain breaks). With LoFTR anchors
the shape-aligned error is ~12 m median; with weaker classical anchors it degrades to ~87 m —
quantifying that *anchor quality*, not the VO mechanism, is the limiter. *(Figure 2: coverage
~6 % single-frame vs 100 % VO-anchored.)*

**6.3 Against the partner's bar.** Adrián cited 10 m across a range of videos as "amazing";
our best single-video accuracy is **~1.8 m** and RoMA reaches **~1.5 m at full coverage** on
random frames — comfortably inside that bar for this clip.

---

## 7. Conclusions

**Technical summary.** Frame↔satellite matching gives **drift-free absolute** localization at
**sub-2 m** where a frame registers; the binding constraint is **coverage** (oblique frames),
which **RoMA largely removes** (≈100 % on a random sample). A VO layer provides a continuous,
shape-faithful trajectory between absolute fixes. The system is **telemetry-free** and
**terrain-agnostic by construction** (deep matching + pluggable provider/matcher), satisfying
the partner's "generic framework" objective.

**Critical analysis / limitations.** Results so far are on a **single clip**; cross-terrain
generalization is **not yet validated** (next: SATLOC, UAV-VisLoc). **RoMA is compute-heavy**
(GPU/Docker) — a real deployment constraint, and it needs a **relative** trust gate rather than
an absolute inlier threshold. Some matchers (LoFTR) mis-read terrain-elevation slope as camera
tilt. Map **currency** is essential at deployment. Heading is recovered but not optimised.

**Impact & business practicality.** The system enables drone navigation wherever GNSS is
denied — search-and-rescue, infrastructure inspection, agriculture, and defense — using only a
camera and a current satellite map. Its modular, config-driven, provider-agnostic design ports
to a new area by swapping the reference source, which is exactly what makes it deployable beyond
the test flight.

**Future work.** (1) RoMA as the production matcher + relative-margin gate; (2) multi-dataset
benchmarking with per-context **auto-selection** of the best matcher; (3) tighter **RoMA + VO
fusion**; (4) GPU deployment for real-time operation. *(Roadmap: `docs/ACCURACY_PLAN.md`.)*

---

## 8. References
`[TODO: format to the citation style required; verify venues/years.]`
- Lowe, D. G. (2004). *Distinctive Image Features from Scale-Invariant Keypoints* (SIFT). IJCV.
- Sun, J. et al. (2021). *LoFTR: Detector-Free Local Feature Matching with Transformers.* CVPR.
- Wang, Y. et al. (2024). *Efficient LoFTR.* CVPR.
- Edstedt, J. et al. (2024). *RoMa: Robust Dense Feature Matching.* CVPR.
- He, X. et al. (2025). *MatchAnything: Universal Cross-Modality Image Matching* (zju3dv).
- Barath, D. et al. (2020). *MAGSAC++.* CVPR.
- Oquab, M. et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision.*
- SATLOC dataset; UAV-VisLoc dataset (partner-referenced). `[TODO: exact citations.]`

## Appendix — reproducibility
```bash
pip install -e ".[dev]"          # core + tests (Python ≤3.12 for deep matchers)
pytest -q                        # 79 offline tests
python scripts/01_extract_frames.py --probe
python scripts/06_extract_gps_track.py
python scripts/07_validate.py --frames 342,3083,6510 --method loftr --provider pnoa
python scripts/08_vo_trajectory.py --provider pnoa --anchors 6400,6500,6600
```
Companion docs: `ACCURACY_LOG.md` (measured results), `docs/ACCURACY_PLAN.md` (roadmap),
`ASSIGNMENT.md` (brief + partner clarifications), `explained-dronomy.md` (walkthrough),
`MERGE_DECISION.md` (architecture rationale).
