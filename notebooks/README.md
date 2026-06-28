# notebooks/ — exploration

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
