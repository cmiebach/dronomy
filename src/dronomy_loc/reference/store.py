"""Persist and reload a fetched GeoImage (PNG + sidecar bbox) so the reference
tile can be reused across scripts without refetching."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .geo import GeoImage


def save_reference(geo: GeoImage, out_dir: str | Path, name: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f"reference_{name}.png"
    Image.fromarray(geo.image).save(img_path)
    np.save(out_dir / f"reference_{name}_bbox3857.npy", np.array(geo.bbox))
    return img_path


def load_reference(out_dir: str | Path, name: str) -> GeoImage:
    out_dir = Path(out_dir)
    img_path = out_dir / f"reference_{name}.png"
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise FileNotFoundError(
            f"{img_path} not found — run scripts/02_fetch_reference.py first.")
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    bbox = tuple(float(v) for v in np.load(out_dir / f"reference_{name}_bbox3857.npy"))
    return GeoImage(image=img, bbox=bbox)
