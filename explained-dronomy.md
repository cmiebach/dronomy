# Dronomy, explained — what we built, why, and how it works

This is the long-form walkthrough of the `diego` branch: every stage of the
pipeline, the reasoning behind each decision, the math underneath, and the
measured results. Written so you can defend any piece of it in a review.

---

## 1. The problem

A drone flies with a camera pointing straight down (**nadir**) and records
video. We get the video and roughly where it was taken — nothing else. No GPS
stream as input, no markers on the ground, no changes to the environment.

**Task: for each video frame, compute where the drone was (latitude,
longitude) and which way it was heading (yaw).**

The only other thing we're allowed is **satellite imagery**, because the brief
is really an *image registration* problem in disguise: the drone frame and the
satellite map show the same patch of Earth, photographed years apart, from
different heights, sensors, seasons and lighting. If you can find how one
image maps onto the other, and you know exactly where every satellite pixel
sits on Earth, then you know where the drone was.

```
drone frame ──► [matching] ──► frame↔satellite homography ─┐
satellite tile (georeferenced) ────────────────────────────┴─► (lat, lon, yaw)
```

Everything in this repo serves that one line.

---

## 2. Step 0 — Understand the data before building anything

### 2.1 The filename lied about which country we're in

The video is `IE_Challenge_lat43_521955_lon5_624290.MP4`. Read naively, that's
(43.521955, **+**5.624290) — southern France. It's actually
(43.521955, **−**5.624290) — **Asturias, northern Spain**. The filename format
simply dropped the minus sign. Building on the wrong sign means fetching
satellite imagery of the wrong country; nothing downstream can ever work. The
lesson is baked into the config as a loud comment, and the prior is stored
with the correct sign.

### 2.2 The video carries its own ground truth (and more)

DJI drones embed a telemetry stream (`djmd`) inside the MP4 container.
`exiftool -ee` decodes it: **6,853 samples — one per video frame** — each with
GPS latitude/longitude, altitude, drone attitude, and gimbal angles.

Two rules about this data:

- **It is ground truth, never input.** The whole point is GPS-denied
  localization; feeding the GPS track into the localizer would be circular.
  We use it exclusively to *score* our estimates (how many metres off were
  we?). This is enforced in module docstrings and the code structure: the
  localizer API simply has no parameter for it.
- **It settles arguments with data.** Decoding the *full* stream (not just
  GPS) produced three load-bearing facts:

| Finding | Value | Consequence |
|---|---|---|
| Gimbal pitch | **−90.0° on every single frame** | The camera was perfectly nadir all flight. A teammate's theory that most frames fail to match because the camera was tilted is **falsified** — the failures must come from *appearance* (vegetation, season gap, blur), not geometry. |
| Relative altitude | ≈ 50 m constant (median 50.0) | Ground footprint per frame is nearly constant (~71 m measured). The search doesn't need to try four tile scales — one suffices (1.8× speedup, measured). |
| Flight envelope | 114 m × 209 m, never > 109 m from the filename prior | ONE 600 m satellite tile covers every possible position. Fetch it once, crop candidates locally, zero further network calls. |

### 2.3 Why frame index = telemetry sample index matters

Every artifact in the pipeline (extracted frames, GPS track, validation rows)
is keyed by the **video frame index**. One telemetry sample per frame means
`gps_track.csv` row *k* is the truth for frame *k* — a clean join key with no
interpolation. Code that parses the telemetry is paranoid about preserving
this alignment (an off-by-one there would silently corrupt every error
measurement we ever make).

---

## 3. Step 1 — Ingest the video so it can never lie to you

`src/dronomy_loc/data/ingest.py` · `scripts/05_ingest_video.py`

### 3.1 Why not just dump frames with OpenCV?

A naive extractor (read frame, save JPEG, repeat) has failure modes that
*silently poison* the dataset: a crash half-way leaves an unknown subset on
disk; a full disk truncates JPEGs that still half-decode; re-running mixes old
and new outputs. For a 3.76 GB / 6,853-frame video where every later accuracy
number depends on these frames, "probably fine" isn't good enough.

### 3.2 The design

1. **Sampling.** One frame per second of video. Decoding all 6,853 4K frames
   to keep 229 sounds wasteful, but it buys the next feature:
2. **Blur filtering (sharpest-in-window).** Instead of taking *the* frame at
   each 1-second mark, we score every frame in the window with the
   **variance of the Laplacian** (a standard focus measure: the Laplacian
   responds to edges; motion blur smears edges; low variance = blurry) and
   keep the sharpest. No fragile absolute threshold — we only ever *rank*
   frames within a window. Sharper frames → more stable features → better
   matching and less VO drift.
3. **Sharding.** Frames are grouped into 30-second shards
   (`shards/shard_0000/...`). The shard is the unit of resume and repair.
4. **The manifest (`manifest.json`).** After each shard completes, a manifest
   is committed **atomically** (write temp file, `os.replace` — the OS
   guarantees you see either the old or the new manifest, never a torn one).
   It records the video's identity (name, size, frame count), every setting,
   and per frame: index, timestamp, blur score, byte size, **SHA-1 hash**.
5. **Verified writes.** Every JPEG is written via
   `imencode → tofile → re-read → hash-compare → decode-check`. (Plain
   `cv2.imwrite` fails *silently* on Windows non-ASCII paths — a real
   gotcha we engineered around.)
6. **Resume & repair.** Re-running replays the same deterministic pass and
   skips shards already marked complete. `--verify` re-reads every frame
   against its recorded hash; damaged shards get demoted to "partial", and
   the next run rebuilds exactly those. An interrupted, corrupted, or
   half-deleted ingest converges back to a verified-complete state by just
   re-running.
7. **`frames.csv`.** A flat index of all kept frames (with blur scores) —
   the table downstream steps join against the GPS track.

Settings changes are detected by hashing the settings dict: you cannot
accidentally mix frames extracted at different resolutions in one directory
(`IngestMismatchError`; `--force` wipes and restarts).

### 3.3 What the real run produced

8 shards, 229 blur-filtered frames from the full 3.8-minute video,
`--verify` clean, and a second run that wrote 0 frames and skipped 229 —
resume short-circuit working on real data.

---

## 4. Step 2 — Georeferenced satellite imagery (the map side)

`src/dronomy_loc/reference/` · `scripts/02_fetch_reference.py`

### 4.1 The key idea: request the bbox yourself, and pixel↔Earth is exact

Satellite tiles are only useful if you know *precisely* where each pixel sits
on Earth. Instead of parsing GeoTIFF metadata (heavy GDAL dependency), we
invert the problem: **we choose the bounding box** and ask a WMS/REST service
to render exactly that box at exactly N×N pixels. The returned raster's
georeferencing is then *by construction*:

- Web-Mercator (EPSG:3857) forward/inverse transforms are ~6 lines of math
  (`geo.py`): `x = R·lon_rad`, `y = R·ln(tan(π/4 + lat_rad/2))` and the
  inverse. Earth radius R = 6378137 m.
- A `GeoImage` couples the raster with its mercator bbox. Pixel→mercator is a
  linear map (note the y-flip: image row 0 is the *north* edge, mercator y
  grows northward); mercator→lat/lon is the inverse formula. So
  `pixel_to_lonlat()` and `lonlat_to_pixel()` are exact, tested round-trips.

One subtlety we fixed: **mercator metres are not ground metres.** The
projection stretches lengths by 1/cos(latitude) — ~1.379× at 43.5°N. Position
output is unaffected (the inverse transform is exact), but anything reporting
a *scale* (our ground-metres-per-pixel altitude proxy) must multiply by
cos(lat). This was found in review and fixed; without it every scale we
reported would be 38% too large.

### 4.2 Four providers, one interface

`provider.fetch(lat, lon, span_m, pixels) -> GeoImage` — pluggable so we can
swap imagery sources without touching anything else:

| Provider | What | Why it's here |
|---|---|---|
| **esri** (default) | Esri World Imagery REST export | Keyless, global, sub-meter. Works with zero setup — removes the Google-auth blocker entirely. Has retry-on-5xx (the endpoint throws sporadic 500s). |
| **pnoa** | Spanish IGN PNOA-MA orthophoto WMS | The *best* source over the flight area: ~0.10–0.25 m/px national orthophoto, keyless. All accuracy results use it. Also retries transient 502s (observed live). |
| **gee** | Google Earth Engine map tiles (Sentinel-2 composite) | The brief names Google Earth; implemented, needs one-time auth. ~10 m/px — too coarse for matching, kept for compliance/comparison. |
| **ign** | French IGN | Legacy of the filename bug (France). Kept as a cautionary tale; doesn't cover Spain. |

Both new providers surface the classic WMS failure mode loudly: services
return *XML error documents with HTTP 200*; we check the Content-Type and
raise with the body text instead of letting PIL choke on XML bytes.

---

## 5. Step 3 — Matching: finding the same points in two very different images

`src/dronomy_loc/matching/`

### 5.1 The classical baseline: SIFT + ratio test + RANSAC

1. **SIFT** detects keypoints and describes the image patch around each with
   a 128-D vector, designed to be invariant to scale and rotation — essential
   because the drone's altitude (scale) and heading (rotation) vs the
   north-up satellite tile are unknown.
2. **Matching + Lowe ratio test.** For each drone descriptor, find its 2
   nearest satellite descriptors; accept only if the best is clearly better
   than the runner-up (distance ratio < 0.75). This kills ambiguous matches
   on repetitive texture (grass, roof tiles).
3. **RANSAC homography.** Even after the ratio test many matches are wrong.
   RANSAC repeatedly fits a homography to random minimal subsets (4 points)
   and counts how many other matches agree within 5 px; the best consensus
   wins, outliers are discarded. The surviving matches are the **inliers** —
   the single most informative number in this whole system.

### 5.2 The deep matcher: LoFTR

SIFT needs *corners*; our scenery is mostly grass and trees, photographed in
a different season than the orthophoto. **LoFTR** is detector-free: a
transformer matches dense coarse-to-fine feature grids between the two
images, finding correspondences even in low-texture regions and across
appearance gaps. It runs through kornia with pretrained "outdoor" weights
(~44 MB, auto-downloaded), CPU-only here at ~3.2 s per image pair.

### 5.3 Why both?

The brief requires comparing ≥2 approaches — but it's also genuinely
instructive: on the bench (Section 9) SIFT produces *confidently wrong*
answers on repetitive structure (133 inliers, 90 m off — the keypoints all
collapse onto a visually repeating pattern), while LoFTR produces *honestly
few* matches that, when they do exceed the trust gate, are metre-accurate.
That asymmetry drives the whole confidence design.

---

## 6. Step 4 — From homography to coordinates

`src/dronomy_loc/localize/pipeline.py`

A homography H maps drone-frame pixels to satellite-tile pixels. From it:

1. **Position**: push the frame's centre pixel through H → a tile pixel →
   `GeoImage.pixel_to_lonlat` → **(lat, lon)**. Nadir camera ⇒ the frame
   centre is (approximately) the point straight under the drone.
2. **Yaw**: push the centre and the pixel one step "up" from centre through
   H; the direction of that vector on the (north-up) tile, measured via
   `atan2(dx, −dy)` (y flipped because image rows grow downward), is the
   heading: 0° = north, 90° = east.
3. **Scale**: the length of one frame pixel on the tile × tile metres-per-px
   × cos(lat) = **ground metres per drone pixel** — a proxy for altitude
   (measured ~0.037 m/px ⇒ ~71 m footprint ⇒ consistent with the 50 m AGL
   telemetry says, given the lens).

---

## 7. Step 5 — The grid search (why matching against one big tile fails)

`src/dronomy_loc/localize/search.py`

### 7.1 The scale-gap problem

A drone frame at 50 m altitude sees ~71 m of ground. A single 1.5 km
reference tile at 4096 px is 0.37 m/px; the same ground in the frame is
~0.037 m/px — a 10× scale gap, plus 200× more area for false matches to hide
in. Matching frame-to-big-tile mostly locks onto off-centre repetitive
structures (measured ~80–90 m biases). The fix:

### 7.2 Grid-of-centres × scales, best-by-inliers

Generate candidate tile centres on a square grid around the coarse prior
(spacing in *projected* metres via mercator offsets), at one or more tile
spans (scales). For each (centre, scale): fetch/crop that small tile, match,
estimate pose, record inliers. **The candidate with the most RANSAC inliers
wins.** Each candidate is exception-isolated — one failed fetch records a
failed candidate instead of killing the search.

### 7.3 The lock gate: when do we believe ourselves?

Empirically (both branches agree): wrong tiles produce **4–9 LoFTR inliers**
(the noise floor); genuinely matchable frames produce **35–117+**. Nothing
lives between ~15 and ~35. So the rule is simple and sharp:
**≥ 20 inliers = locked (trust it); below = report nothing.**
Precision over recall: a wrong answer that looks confident is worse than no
answer, because downstream (VO anchoring, report claims) builds on locks.

### 7.4 The one-world-tile optimization

Because the whole flight fits within ~109 m of the prior (telemetry finding
3), we fetch ONE 600 m PNOA tile at 4096 px (0.146 m/px) and serve every
candidate as a **local crop** (`make_world_fetch` in `validate.py`): bbox →
world pixels → slice → resize → `GeoImage` with the cropped bbox. 100
candidates = 1 network call total. A `TileCache` keyed by
(lat, lon, span, pixels) removes even repeated crops across frames.

### 7.5 The single-scale speedup

Telemetry finding 2 (constant altitude) says the footprint never changes, so
the 4-scale sweep is redundant. Searching a single 70 m scale on a denser
grid produced the *identical lock outcome* on the bench at **1.8× less
compute** — and the frames that couldn't be trusted now produce *no pose at
all* rather than sub-gate noise. (The localizer never reads the altitude
value itself — the scale is calibrated from a locked match — so this stays
telemetry-free.)

---

## 8. Step 6 — Measurement: harness, bench, and honest numbers

`src/dronomy_loc/localize/validate.py` · `scripts/07_validate.py` ·
`src/dronomy_loc/data/telemetry.py` · `scripts/06_extract_gps_track.py` ·
`ACCURACY_LOG.md`

### 8.1 Ground-truth extraction

`telemetry.py` wraps exiftool: run `-ee -j -n -G3`, parse the `Doc<N>:`
groups (frame = N−1, *kept even when fixes are dropped* so indices never
shift), validate fixes ((0,0) and out-of-range dropped, counted), write
`gps_track.csv`. The subprocess handling kills the **whole process tree** on
timeout — on Windows, exiftool is a launcher whose perl child inherits the
pipes; killing only the parent leaves the script hung forever (a classic).

Our extraction agrees with the teammate's independent one to **0.00 m** on
the bench frames — two implementations, same truth.

### 8.2 The validation harness

`validate_frames` takes a set of frames (explicit list or "N spread evenly"),
localizes each from the same coarse prior, joins against the GPS track, and
reports: per-frame error (haversine metres), lock rate, and median/mean/worst
computed **over locked frames only** — unlocked frames are reported as
exactly that, not averaged into a misleading blend. Results go to a CSV with
an exact-round-trip reader for plotting.

### 8.3 The frozen bench

Three frames (342 / 3083 / 6510), fixed prior, fixed grids, fixed source —
*identical* to the teammate's bench so every number is directly comparable
across branches. 6510 is the interesting one: the documented "degenerate
SIFT" case. Frozen means frozen: any change to the protocol is a new named
bench, otherwise comparability dies.

---

## 9. Measured results (all on CPU, PNOA source, real video)

### 9.1 The three-frame bench

| Config | 342 | 3083 | 6510 | Mean | Teammate's |
|---|---|---|---|---|---|
| SIFT, his grid | 20.0 m (39 inl) | no pose | 90.7 m (133 inl ⚠ false confidence) | **55.3 m** | 56.8 m ✓ |
| LoFTR, his grid | 106.7 m (15 inl → unlocked) | 94.7 m (15 inl → unlocked) | **1.76 m (117 inl, LOCKED)** | **67.7 m** | 70.0 m ✓ |
| LoFTR, single-scale 70 m (ours) | no pose | no pose | **1.75 m (35 inl, LOCKED)** | locked-only 1.75 m | — (1.8× faster) |

Reading it:
- Reproducing his numbers within noise **validates the entire pipeline** —
  ingest, geo math, providers, search, scoring — end to end on real data.
- The SIFT 6510 row is the cautionary tale: 133 inliers *and* 90 m wrong.
  Inlier count from SIFT on repetitive structure is not confidence.
- The LoFTR 6510 row is the proof of concept: **metre-level absolute
  localization from video alone**, where the scene allows it.
- The "ours" row shows the telemetry-informed simplification costs nothing
  and runs 1.8× faster (153 s vs 278 s per frame).

### 9.2 Why most frames don't lock (and what we did about it)

Only a small fraction of this flight matches the orthophoto directly — the
teammate measured ~6% (a segment near the flight's end, frames ~6400–6600).
His explanation (tilted camera) is disproven by our telemetry decode; the
true cause is **appearance**: feature-poor vegetation photographed in a
different season than the reference. You cannot match texture that isn't
there — so instead of fighting it, we route around it:

---

## 10. Step 7 — VO dead-reckoning: coverage beyond the matchable segment

`src/dronomy_loc/localize/odometry.py` · `scripts/08_vo_trajectory.py`

### 10.1 The idea

Consecutive video frames overlap enormously (at 3 fps and ~1 m/s, ~99%
overlap) and *always* match each other — same sensor, same second, same
lighting. Over near-planar ground with a nadir camera, the motion between
consecutive frames k−1 and k is a homography H₍k−1→k₎.

If some frame *a* (an **anchor**) also has a verified absolute registration
H₍a→ref₎ to the satellite tile (a locked grid-search match), then ANY frame k
can be georeferenced by composing along the chain:

```
H(k→ref) = H(a→ref) · H(a−1→a)⁻¹ · … · H(k→k+1)⁻¹      (chain k → a → tile)
```

and the standard pose extraction (Section 6) applies unchanged. After every
multiply we renormalize H by H[2,2] so numbers stay conditioned over long
chains.

### 10.2 Drift, breaks, and anchors

- **Drift**: each pairwise homography has sub-pixel error; composing N of
  them accumulates error roughly with chain length. So the deliverable is an
  **error-vs-hops-from-anchor curve**, not a flat accuracy claim.
- **Breaks**: a frame that matches neither neighbour (e.g. pure blur) splits
  the chain; frames beyond it are unreachable from that side's anchor and are
  *omitted* (honesty again — no estimate rather than a fabricated one).
- **Anchors reset drift**: every additional locked frame becomes an anchor;
  each frame chains to its *nearest* anchor by hop count. More matchable
  moments ⇒ flatter error curve.
- Note the distinction from the teammate's "fusion is not viable" verdict:
  that referred to *triangulating between multiple independent fixes* (true,
  there's only one matchable segment). Odometry chaining needs only ONE
  anchor — different mechanism, different feasibility.

### 10.3 The real-data run

Anchors: grid-search LoFTR locks at keyframes in the matchable segment
(telemetry-free — prior is the filename coordinate). Sweep: every 10th frame
(686 frames at ~3 fps), SIFT pairwise links (consecutive frames are easy —
SIFT is fine and fast here), then chain and score every swept frame against
the GPS track.

**Measured results (2026-06-10, real video, all telemetry-free):**

- All three anchor keyframes locked: 6400 at 2.54 m, 6500 at 1.70 m,
  6600 at 1.59 m (33–57 inliers each).
- The chain held across the **entire flight**: 685/685 links, zero breaks.
- **Coverage: 686/686 swept frames = 100%** — against the ~6% ceiling of
  per-frame matching.
- Full-trajectory error vs all GPS fixes: **median 12.3 m, RMSE 35.7 m,
  worst 70.2 m**.
- The drift curve behaves exactly as theory predicts: median **1.6 m**
  within 10 hops of an anchor, 4.2 m at 11–50 hops, 30.3 m beyond (the far
  end of the flight is ~640 hops from the nearest anchor, since all three
  anchors sit in the final matchable segment). Every additional anchor
  flattens that tail.
- Honest caveats: this flight is slow (~1 m/s) and the scene static, which
  flatters VO; and the GT itself is consumer GPS (±1–3 m). The
  error-vs-hops curve in `data/outputs/vo_trajectory.csv` is the honest
  deliverable, not the headline mean.

---

## 11. Engineering discipline (the part that doesn't show in demos)

- **Every module has offline tests** (56 passing): synthetic videos generated
  with `cv2.VideoWriter`, a synthetic textured "world" with exact known
  geometry for search/VO tests (ground truth by construction — the search
  test recovers a planted frame to 0.009 m), network mocked at the module
  boundary, exiftool mocked at the subprocess boundary. `pytest` runs in ~30 s
  with no GPU, no network, no video — so it actually gets run.
- **Adversarial review before pushing.** Independent reviewers hunted the new
  code; each non-minor claim was then *re-verified by a separate skeptic with
  a repro*. Three confirmed real bugs got fixed + regression-tested:
  1. resume with a constant `--max` cap made zero progress (livelock);
  2. `frames.csv` could be lost forever in a crash window after completion;
  3. the 1.379× mercator/ground scale error (Section 4.1).
- **Determinism**: sampling replay, tie-breaks, cache keys, test seeds — all
  fixed, so two runs of anything agree.
- **Git hygiene**: the 3.76 GB video, frames, tiles, tracks and outputs are
  all gitignored; the repo holds only code, config, docs and tests. GitHub is
  source of truth for code, never data.
- **ASCII-only runtime output** (Windows consoles mangle em-dashes under
  cp850/cp1252 — observed live, then swept from every print).

---

## 12. Current state & what's next

**Done and measured**: ingest (8 shards / 229 verified frames), ground truth
(6,853 fixes), four imagery providers (two keyless verified live), grid
search with lock gate, three-frame bench reproducing the teammate's numbers,
metre-level locks on the matchable segment, VO modules tested, full-trajectory
VO run.

**Next**:
1. **Densify anchors** in/around frames 6300–6700 (the more locks, the
   flatter the VO drift curve) and extend the swept range.
2. **Appearance-gap experiments** (the 6%-coverage attack): PNOA acquisition
   date selection, CLAHE/gradient pre-normalization, cross-modal matchers
   (MatchAnything; the teammate has a Docker image to reuse).
3. **35-stop matchability scan** with our pipeline to confirm the coverage
   map frame-by-frame against his.
4. **Report**: framing already settled by the data — "metre-level where the
   scene permits; VO-chained, drift-bounded estimates elsewhere; coverage is
   an appearance problem, quantified."
5. **Professor decisions pending**: telemetry purity for engineering priors
   (we self-calibrate regardless), grading scope on coverage, which number is
   the graded number. (Note: GT itself is consumer GPS, ±1–3 m — sub-metre
   claims are unprovable against it.)
