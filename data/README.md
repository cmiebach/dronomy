# data/ — generated artifacts (git-ignored)

This directory holds everything produced at runtime. Contents are **not** committed
(see `.gitignore`); the subfolders are created automatically by the scripts.

- `frames/`    — sampled frames from the drone video (`01_extract_frames.py`)
- `reference/` — fetched georeferenced satellite tiles + `*_bbox3857.npy` sidecars (`02_fetch_reference.py`)
- `outputs/`   — overlays, trajectory CSV, and trajectory plots (`03/04`)
- `cache/`     — scratch
