# Literature review — UAV visual geo-localization (working notes)

Survey of image-registration / matching methods for matching nadir drone frames
to georeferenced satellite imagery. (Workflow step 1 from the meeting notes.)

## Problem framing
Nadir drone frame ↔ orthorectified satellite tile is a **2D image registration**
problem well-modeled by a **homography / similarity transform** (both views are
top-down). Main nuisances: scale (altitude), rotation (heading), and **cross-domain
appearance** differences — season, lighting, sensor, and recency.

## Classical feature-based matching
- **SIFT** — scale/rotation invariant; the canonical baseline (brief names it).
- **ORB / AKAZE** — faster, binary descriptors; weaker cross-domain.
- Pipeline: detect+describe → ratio-test match → RANSAC homography.
- Strength: no training, interpretable. Weakness: degrades on low-texture
  (grass/fields — exactly our scene) and large appearance gaps.

## Learned matching (modern)
- **SuperPoint + SuperGlue** — learned keypoints + attention-based matcher.
- **LoFTR** — *detector-free*, dense coarse-to-fine; strong in low-texture/repetitive
  regions → promising for our grass-heavy footage. (kornia ships pretrained weights.)
- **DISK, ALIKED, RoMa/DKM** — newer dense/robust matchers worth a mention.
- Strength: robust cross-domain. Weakness: compute (GPU ideal), weights/setup.

## Cross-view / retrieval (context, likely out of scope)
- Dedicated UAV-vs-satellite geo-localization datasets/methods (e.g. University-1652)
  target *oblique* cross-view retrieval. Our nadir case is simpler — note for the report.

## Evaluation
- No ground-truth trajectory yet (only the filename coordinate, ~takeoff/center).
- Qualitative: footprint/overlay correctness, trajectory smoothness & plausibility.
- Quantitative (if GT obtained): position error (m) via haversine, yaw error,
  per-frame success rate, runtime per frame. Compare SIFT vs LoFTR on these.

## To compare (the brief's "≥2 approaches")
| Axis | SIFT (classical) | LoFTR (deep) |
|---|---|---|
| Inliers on grass | ? | ? |
| Success rate over video | ? | ? |
| Runtime/frame (CPU) | ? | ? |
| Setup cost | none | torch+weights |

(Fill in from `notebooks/03_matcher_comparison.ipynb`.)

## References
_TODO: add citations (SIFT 2004; SuperGlue 2020; LoFTR 2021; RoMa 2023; survey papers)._
