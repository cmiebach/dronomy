# data/outputs/ — committed result artifacts

Generated outputs are normally gitignored. We deliberately commit a small set of
**result CSVs** here so the numbers cited in the report and poster can be verified
directly from a clone. Regenerating them requires a GPU and the source video
(hours of compute), which a reviewer will not have — so without these files the
cited figures would not be reproducible from the repository alone.

These are small summary results (largest ~70 KB), not raw data. Rasters, figures,
and the video remain gitignored. Ground truth (GPS) is used only to score results;
it is never an input to localization.

| File | What it backs |
|---|---|
| `full_pipeline_roma.csv` | Blind whole-video RoMa pipeline: median 7.4 m, 55% within 15 m, 28/28 anchors |
| `vo_trajectory.csv` | Visual odometry: raw median 23.2 m, shape-aligned RMSE 137 m, path-length ratio 3.11 (686 frames) |
| `validation_loftr_mps.csv` | LoFTR blind scan: ~15% coverage, median 2.6 m, best 1.8 m (40 frames) |
| `val_loftr_12.csv`, `val_sift_12.csv`, `val_roma_12.csv` | Per-frame LoFTR / SIFT / RoMa validation (12-frame sample) |
| `E2E_NUMBERS.md` | End-to-end run summary |
