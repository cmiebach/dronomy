# Report outline (draft)

Final deliverables: **code (GitHub) + written report + presentation**. Structure to
be confirmed with Adrian (Dronomy CEO) — lock that meeting early.

1. **Introduction** — GPS-denied navigation; why it matters for Dronomy (indoor → outdoor).
2. **Problem statement** — inputs (nadir video + rough GPS), output (absolute pose),
   constraints (no GPS, no environment alteration).
3. **Data** — the video (4K, 30 fps, ~229 s), scene description, location, metadata/
   altitude check, reference-imagery sources (Google Earth vs IGN open data).
4. **Related work / methods surveyed** — see `docs/literature-review.md`.
5. **Approach**
   - Reference fetch & georeferencing (pixel ↔ lat/lon).
   - Frame extraction.
   - Matching: classical (SIFT) vs deep (LoFTR) — the required comparison.
   - Pose from homography (position, yaw, scale).
6. **Experiments & results** — matcher comparison table, qualitative overlays,
   trajectory plot, failure cases (low-texture grass, occlusion).
7. **Bonus** — visual odometry + fusion (if attempted).
8. **Limitations & future work** — GT for evaluation, real-time, robustness.
9. **Conclusion**.

## Open questions for Adrian
- Expected accuracy / evaluation criteria?
- Is altitude/telemetry (e.g. an `.SRT`) available to use or as ground truth?
- Must Google Earth specifically be used, or is open satellite data acceptable?
- Real-time expectations for the final demo?
