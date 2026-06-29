# notebooks/ — exploration

## ⭐ `dronomy_end_to_end.ipynb` — the single end-to-end notebook
One notebook that runs the **whole project** by driving the same scripts the graders use on the
CLI: fetch → ingest → GPS ground truth → reference imagery → localize with **every matcher
(SIFT, LoFTR, RoMA) + per-frame auto-selection** → whole-video **VO trajectory** → **SE(2) shape
metric** → **figure suite**, then checks every headline number **1:1 against `STATUS.md`/the
report** (with tolerances sized to the documented ~±10 m run-to-run drift — small drift is normal
and flagged as a MATCH). Missing deep-matcher deps or GPU are reported as *unavailable* and the run
continues; it never crashes. Run **Cell → Run All**, or headless:
```bash
pip install -e ".[notebook,deep]"
jupyter nbconvert --to notebook --execute notebooks/dronomy_end_to_end.ipynb
```

---

Jupyter notebooks for data exploration, matcher comparison, and figure generation
for the report. Keep heavy/throwaway exploration here; promote anything reusable
into `src/dronomy_loc/`.

Suggested notebooks:
- `01_explore_video.ipynb` — frame samples, scene content, metadata, altitude check
- `02_reference_imagery.ipynb` — compare IGN vs GEE tiles, resolution, recency
- `03_matcher_comparison.ipynb` — SIFT vs LoFTR: inliers, success rate, runtime (the brief's "compare ≥2")
- `04_results.ipynb` — trajectory plots and qualitative overlays for the report

Run with the repo root as CWD so `config.yaml` and `data/` resolve, or
`import sys; sys.path.insert(0, "src")` to import `dronomy_loc`.

## Running `framework_demo.ipynb`
Needs the notebook extras (table + inline image + a kernel):
```bash
pip install -e ".[notebook,deep]"          # pandas + jupyter + (LoFTR) torch
jupyter notebook framework_demo.ipynb
# or headless: python -m nbconvert --to notebook --execute framework_demo.ipynb
```
Verified end-to-end (SIFT, 6 frames) → metrics table, comparison figure,
GeoJSON/KML export, GT-vs-estimate overlay.
