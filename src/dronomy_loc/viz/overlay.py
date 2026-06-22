"""Visualization: side-by-side matches, the drone-frame footprint projected onto
the reference tile, and a trajectory plot. Useful for the qualitative evaluation
the brief expects (no ground-truth track available yet)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..matching.base import MatchResult
from ..reference.geo import GeoImage


def draw_matches(drone_bgr, ref_rgb, mr: MatchResult, max_draw: int = 80) -> np.ndarray:
    """Side-by-side inlier correspondences (drone | reference)."""
    ref_bgr = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGR)
    h = max(drone_bgr.shape[0], ref_bgr.shape[0])
    d = cv2.copyMakeBorder(drone_bgr, 0, h - drone_bgr.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    r = cv2.copyMakeBorder(ref_bgr, 0, h - ref_bgr.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    canvas = np.hstack([d, r])
    off = d.shape[1]
    mask = mr.inlier_mask if mr.inlier_mask is not None else np.ones(len(mr.src_pts), bool)
    idx = np.where(mask)[0]
    for i in idx[:max_draw]:
        p1 = tuple(np.round(mr.src_pts[i]).astype(int))
        p2 = tuple(np.round(mr.dst_pts[i]).astype(int) + [off, 0])
        cv2.line(canvas, p1, p2, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, (0, 200, 255), -1)
        cv2.circle(canvas, p2, 3, (0, 200, 255), -1)
    return canvas


def draw_frame_footprint(ref_rgb, H: np.ndarray, frame_shape) -> np.ndarray:
    """Draw the drone frame's outline (warped by H) on the reference tile."""
    h, w = frame_shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(corners, H).reshape(-1, 2).astype(int)
    out = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGR).copy()
    cv2.polylines(out, [proj], True, (0, 0, 255), 3, cv2.LINE_AA)
    c = proj.mean(axis=0).astype(int)
    cv2.circle(out, tuple(c), 6, (255, 0, 0), -1)
    return out


def plot_trajectory(ref: GeoImage, lats, lons, out_path: str | Path, title: str = "Estimated trajectory"):
    """Overlay the estimated lat/lon track on the reference tile and save a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    minx, miny, maxx, maxy = ref.bbox
    lon0, lat0 = ref.pixel_to_lonlat(0, 0)            # top-left
    lon1, lat1 = ref.pixel_to_lonlat(ref.width, ref.height)  # bottom-right
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(ref.image, extent=[lon0, lon1, lat1, lat0])  # extent in lon/lat
    ax.plot(lons, lats, "-o", color="red", markersize=3, linewidth=1.5, label="estimate")
    if lons:
        ax.scatter([lons[0]], [lats[0]], c="lime", s=60, zorder=5, label="start")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(title); ax.legend()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
