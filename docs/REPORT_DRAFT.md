# GPS-denied Drone Visual Localization — Written Report (DRAFT v0.1)

> **Status: first draft for the team to edit.** Maps 1:1 to the capstone report
> brief (Objective → Data → Methodology → Potential Solutions → Development →
> Conclusions) + ethics statement on the cover. Final format: 10–20 pages, 1.0
> spacing, 12-pt Times New Roman, PDF. Due 29 June 2026. **`[TODO]`** marks gaps
> teammates must fill (names, final multi-dataset numbers, figures).

---

## Cover page

**Title:** Telemetry-free Visual Localization of Drones in GNSS-denied Environments
**Programme:** IE MBDS Capstone Project · Partner: **Dronomy**
**Team (Group 3):** `[TODO names]` · **Academic tutor:** Hind `[TODO]` · **Company mentor:** Adrián Carrio (Dronomy)
**Date:** June 2026

**Ethics statement** *(required on the cover):*
> This project develops absolute drone localization for **GNSS-denied environments**,
> an explicitly dual-use capability (search-and-rescue, infrastructure inspection,
> precision agriculture, and defense/battlefield navigation). We commit to
> responsible use and note the following. (1) **No personal data:** the system uses
> only aerial terrain imagery and public/commercial satellite basemaps; no
> identifiable individuals are processed. (2) **Telemetry-free by design:** the
> localizer never consumes GPS/telemetry at runtime — GPS is used only offline as a
> scoring reference — so the method does not depend on, or expand, surveillance
> infrastructure. (3) **Imagery licensing:** satellite/orthophoto sources are used
> within their terms (Esri World Imagery, Spanish IGN PNOA open data, Google Earth).
> (4) **Dual-use acknowledgment:** we recognise the defense applications discussed
> with the partner and present the work for civilian-protective and humanitarian
> contexts. `[TODO: team to confirm wording with tutor.]`

---

## 1. Objective (problem & aim)

**Problem.** Most commercial drones navigate with GPS/GNSS, but GNSS is unavailable,
degraded, or jammed in many real settings (indoor-to-outdoor transitions, dense
terrain, conflict zones). Dronomy builds GPS-denied autonomous flight; this project
targets the **outdoor** case.

**Aim.** Given (a) aerial RGB video from a drone's **bottom-looking (nadir) camera**
and (b) a **georeferenced satellite map** of the area, estimate the drone's
**absolute pose** — latitude/longitude (required) and heading (secondary) — for each
frame, **without any GPS/telemetry input at runtime**.

**Success criteria (from the partner, Adrián, 2026-06-23).** The deliverable is a
**generic, robust framework** that works across varied terrain (forest, rivers,
rocks, open fields), *not* a solution tuned to one video. **Code quality and
generality outweigh per-frame accuracy.** Reference bar: "10 m average error across a
range of videos would be amazing." Position accuracy matters more than heading;
trajectory **shape/dimensions** matching the true path (precision) is valued over a
small absolute offset.

---

## 2. Data sources

**2.1 Drone footage (input).** One DJI **Mavic 3 Enterprise (M3E / WM265E)** clip:
4K **3840×2160**, **29.97 fps**, 6 853 frames, 228.7 s, H.264; bottom-looking wide
camera; recorded over **Asturias, Northern Spain** (43.5220, −5.6243). At runtime the
**only input is the image stream** (partner-confirmed).

**2.2 Ground truth (scoring only — never a model input).** The MP4 embeds DJI
telemetry as a protobuf metadata stream (`dvtm_wm265e.proto`). We decode a
**per-frame GPS + altitude** track for all 6 853 frames with `exiftool -ee`
(`GPSLatitude/Longitude`, `AbsoluteAltitude`, `RelativeAltitude`). It shows a
near-constant cruise altitude of **~49 m AGL** (take-off 6.6 m → ~50 m). Used solely
to measure error.

**2.3 Reference satellite imagery (the map).** Pluggable providers, all returning a
georeferenced tile with exact pixel↔lat/lon mapping:
- **Esri World Imagery** — keyless, global, sub-meter; default.
- **Spanish IGN PNOA orthophoto** — keyless WMS, **~0.15–0.25 m/px** over Asturias; best resolution here.
- **Google** (Static Maps / Earth / Earth Engine) — partner-suggested; manual single-map export acceptable.
- *Map currency is mandatory* (deployment is always "now"); seasonal matching was considered and **dropped** (partner: recency wins).

**2.4 Camera intrinsics.** Identified from the container (`encoder=DJI DJIMavic3
Enterprise`) and partner reference image. M3E wide: 4/3 CMOS 17.3×13 mm, 24 mm-equiv,
DFOV 84° (photo). **Focal length ≈ 3 713 px** (partner, at 5 280 px photo width) →
**≈ 2 700 px at the 3 840 px video** — used for scale recovery / footprint (~70 m
across at 49 m AGL, GSD ~1.8 cm/px). `[TODO: pin exact CalibratedFocalLength +
principal point from the reference still's DJI XMP.]`

**2.5 Generalization datasets (planned).** **SATLOC** and **UAV-VisLoc** (partner-
recommended) for cross-terrain validation. `[TODO: ingest + GT availability check.]`

---

## 3. Methodology

**3.1 Pipeline (framework).** A config-driven, plug-and-play pipeline:
`ingest video → extract/verify frames → fetch georeferenced reference → match frame
↔ map → homography → pose (lat/lon/yaw/scale) → grid search + lock gate → validate
vs GPS → (optional) visual-odometry fusion`. Implemented as the installable package
`dronomy_loc` with sub-modules `data/`, `reference/`, `matching/`, `localize/`,
`viz/`, driven by `config/config.yaml` and runnable via numbered `scripts/01..10`.

**3.2 Frame ingestion.** Sharded (~30 s shards), resumable, **SHA-1 integrity-
verified**, blur-aware frame selection (keep the sharpest frame per window) to keep
out-of-focus frames from poisoning matching.

**3.3 Matching (the core).** A pluggable `Matcher` interface with four backends:
- **SIFT** (classical, Laplacian-of-Gaussian keypoints) — fast, strong same-modality, but appearance- and texture-dependent; weak/degenerate cross-modal.
- **LoFTR** (transformer, detector-free, semi-dense) — handles low texture; moderate cost.
- **eLoFTR** (Efficient LoFTR) — faster LoFTR variant.
- **RoMA / MatchAnything** (dense, DINOv2 backbone) — **state of the art** for cross-modal/oblique matching (partner-endorsed).
Homography is fit with **MAGSAC++** (robust RANSAC). A **grid-of-centres × multi-
scale** search around the coarse prior handles unknown position offset and scale
(unknown altitude); the candidate with the most inliers wins.

**3.4 Confidence / lock gate.** For sparse matchers, an absolute inlier threshold
(~20 = trusted; 4–9 = noise floor) gates a "lock". For **dense** matchers (RoMA),
absolute counts are invalid (wrong tiles also score high), so a **relative-margin
gate** (best vs second-best) is used. `[TODO: finalise margin gate in code.]`

**3.5 Visual odometry (secondary).** Frame-to-frame matching (SIFT, trivial in the
same modality — 5–10 k inliers/pair) + partial-affine motion, integrated into a dense
trajectory; **anchored on satellite-locked frames** to bound drift. Heading is used
only to reject implausible yaw jumps.

**3.6 Evaluation metric.** Beyond per-frame haversine error, a **trajectory-shape
metric**: align the estimated track to GPS with a similarity/rigid transform
(Umeyama/Procrustes) and report the residual RMSE — directly encoding the partner's
"matches the path in shape and dimensions" criterion. 79 offline unit tests cover
geo math, ingestion, providers, search, validation, and VO.

---

## 4. Potential solutions (alternatives considered)

The brief requires ≥2 alternatives with rationale. We evaluated five and chose a
deep-matching + hybrid design.

| Alternative | Idea | Pro | Con | Verdict |
|---|---|---|---|---|
| **A. Classical map-matching (SIFT)** | keypoint match frame↔map | fast, no GPU, no deps | cross-modal/low-texture failures; degenerate locks (frame 6510: 136 inliers, **90 m wrong**) | baseline only |
| **B. Deep map-matching (LoFTR)** | transformer detector-free | locks matchable frames to **~1.8 m** | only ~6 % of (oblique) frames register; tilt-sensitive | strong but low coverage |
| **C. Deep map-matching (RoMA)** ★ | dense DINOv2 cross-modal | **10/10 random frames matchable, ~1.5 m**, 45–310× more inliers than LoFTR; robust to tilt | heavy (GPU/Docker) | **chosen primary matcher** |
| **D. Visual odometry only** | chain frame-to-frame motion | 100 % coverage, no map needed | **drifts** (no absolute reference) | secondary |
| **E. Hybrid (C + D)** ★ | RoMA absolute fixes + VO between | continuous *and* drift-corrected | most engineering | **chosen architecture** |

**Rationale for the choice.** The partner named RoMA as SOTA and prioritised
generality over per-scene tuning; deep matching avoids hand-calibration and
generalises across terrain. RoMA removes the coverage ceiling (≈6 % → ≈100 %) while
holding ~1–2 m precision, and VO fills any gaps. SIFT is retained as a fast baseline
for the benchmarking layer.

---

## 5. Development (code & engineering practices)

- **Package:** `dronomy_loc` (src-layout, `pip install -e .`); clean ABCs
  (`Matcher`, `ReferenceProvider`) + factories; typed `GeoImage` geo-contract.
- **Reproducibility:** central `config/config.yaml`; numbered `scripts/01..10` as
  "small working pieces"; `data/` git-ignored (no data ever in git).
- **Testing:** **79 offline unit tests** (synthetic videos, mocked network; no torch
  needed) — geo, ingest, telemetry, providers, search, validate, VO.
- **Deep/RoMA env:** torch+kornia for LoFTR (Python ≤3.12); RoMA via
  `docker/Dockerfile.matchanything` (zju3dv weights). Missing-dependency runs now
  fail with a clear message rather than silently returning 0 inliers.
- **Robustness:** integrity-verified ingestion; per-tile error tolerance in the
  search; deterministic tie-breaking.
- `[TODO: link the GitHub repo + an accessible notebook (brief requires a notebook).]`

---

## 6. Results & evaluation

*All on the provided clip; reference = PNOA. (Generalization numbers pending — §7.)*

| Method | Matchable frame (6510) | Coverage (random frames) | Notes |
|---|---|---|---|
| SIFT (classical) | 90.7 m (degenerate lock) | low | confident-but-wrong locks |
| **LoFTR** | **1.76–1.80 m** (165 inliers) | ~6 % (oblique frames unlock) | tilt-sensitive |
| **RoMA** | sub-2 m | **10/10 random frames; median 1.5 m, best 0.7 m** | 45–310× LoFTR inliers |
| **VO (anchored)** | — | **100 % trajectory coverage** (686/686) | median ~12 m (LoFTR anchors); shape preserved |

Headline: **~1.8 m best single-frame accuracy** and, with RoMA, **~100 % coverage at
~1.5 m** on random frames across the flight — well inside the partner's "10 m =
amazing" bar (single video). `[TODO: GT-vs-RoMA overlay figure (outputs/ma_bench/
overlay_crop.jpg); SE(2)-aligned trajectory ATE; per-dataset table.]`

---

## 7. Conclusions

**Technical.** Frame↔satellite matching gives drift-free absolute localization with
**sub-2 m** accuracy where a frame registers; the limiter is **coverage** (oblique,
forward-flight frames), which **RoMA largely removes** (≈100 % on a random sample).
VO provides a continuous, shape-faithful trajectory between absolute fixes. The
system is **telemetry-free** and **terrain-agnostic by construction** (deep matching,
pluggable provider/matcher).

**Critical analysis / limitations.** Results so far are on a single clip; **cross-
terrain generalization is not yet validated** (next: SATLOC, UAV-VisLoc). RoMA is
**compute-heavy** (GPU/Docker) — a deployment constraint. Terrain elevation can be
mis-read as camera tilt by some matchers. Dense matchers need a **relative** trust
gate. Map **currency** is essential in deployment.

**Impact & business practicality.** Enables drone navigation where GNSS is denied —
search-and-rescue, infrastructure inspection, agriculture, and defense — using only a
camera and a current satellite map. Modular, config-driven, and provider-agnostic, so
it ports to new areas by swapping the reference source.

**Future work.** (1) RoMA as the production matcher + relative margin gate;
(2) multi-dataset benchmarking with auto-selection of the best matcher per context;
(3) hybrid RoMA+VO fusion; (4) GPU deployment. *(See `docs/ACCURACY_PLAN.md`.)*

---

## Appendix / references
- `[TODO: references — LoFTR, Efficient LoFTR, RoMA/MatchAnything, MAGSAC++, DINOv2, SATLOC, UAV-VisLoc.]`
- Repo docs: `ACCURACY_LOG.md` (measured results), `PLAN.md` / `docs/ACCURACY_PLAN.md`
  (roadmap), `ASSIGNMENT.md` (brief + partner clarifications), `explained-dronomy.md`
  (walkthrough), `MERGE_DECISION.md` (architecture choice).
