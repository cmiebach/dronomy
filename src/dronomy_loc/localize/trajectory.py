"""Trajectory scoring for the *shape-precision* metric Adrian asked for:
"a trajectory similar in shape and dimensions to the original path, even if it
is off the ground truth by a few meters."

Absolute per-frame error punishes a constant offset/heading bias that does not
actually distort the path. The honest metric for "same shape and dimensions" is
the error AFTER a rigid SE(2) alignment (rotation + translation, **no scale** —
scaling would hide a wrong-size path). We report both raw and aligned ATE plus
a path-length ratio, and expose the alignment so the overlay plot can show the
estimated path laid on top of the truth.

Everything works in a local east/north metre plane around a reference latitude
(equirectangular approx — fine over this ~200 m flight); lat/lon <-> metres uses
the same `meters_per_degree_lat` as the rest of the project.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..reference.geo import meters_per_degree_lat


@dataclass
class SE2:
    """Rigid 2-D transform: maps a point p (east, north) -> R @ p + t."""
    R: np.ndarray          # 2x2 rotation
    t: np.ndarray          # 2 translation (metres)
    rot_deg: float         # rotation as a heading offset, for reporting

    def apply(self, pts: np.ndarray) -> np.ndarray:
        return pts @ self.R.T + self.t


@dataclass
class TrajectoryMetrics:
    n: int
    ate_raw_m: float          # RMS abs error, no alignment
    ate_aligned_m: float      # RMS error after rigid SE(2) alignment (the metric)
    mean_aligned_m: float
    median_aligned_m: float
    worst_aligned_m: float
    path_len_est_m: float
    path_len_gt_m: float
    path_len_ratio: float     # est / gt; 1.0 = identical dimensions
    align: SE2


def lonlat_to_local_m(lats, lons, ref_lat: float, ref_lon: float) -> np.ndarray:
    """(lat, lon) arrays -> Nx2 (east, north) metres about (ref_lat, ref_lon)."""
    m_lat, m_lon = meters_per_degree_lat(ref_lat)
    east = (np.asarray(lons) - ref_lon) * m_lon
    north = (np.asarray(lats) - ref_lat) * m_lat
    return np.column_stack([east, north])


def align_se2(src: np.ndarray, dst: np.ndarray) -> SE2:
    """Best rigid transform (rotation+translation, NO scale) taking src->dst,
    least-squares (Umeyama without scaling). src, dst are Nx2."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    sc, dc = src.mean(0), dst.mean(0)
    H = (src - sc).T @ (dst - dc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, d]) @ U.T          # 2x2, proper rotation
    t = dc - R @ sc
    return SE2(R=R, t=t, rot_deg=math.degrees(math.atan2(R[1, 0], R[0, 0])))


def _path_length(pts: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if len(pts) > 1 else 0.0


def score_trajectory(est_lat, est_lon, gt_lat, gt_lon) -> TrajectoryMetrics:
    """Score an estimated track against ground truth. Inputs are equal-length
    sequences ordered by frame (lat, lon in degrees)."""
    ref_lat = float(np.mean(gt_lat))
    ref_lon = float(np.mean(gt_lon))
    est = lonlat_to_local_m(est_lat, est_lon, ref_lat, ref_lon)
    gt = lonlat_to_local_m(gt_lat, gt_lon, ref_lat, ref_lon)

    raw = np.linalg.norm(est - gt, axis=1)
    align = align_se2(est, gt)
    res = np.linalg.norm(align.apply(est) - gt, axis=1)

    return TrajectoryMetrics(
        n=len(gt),
        ate_raw_m=float(np.sqrt((raw ** 2).mean())),
        ate_aligned_m=float(np.sqrt((res ** 2).mean())),
        mean_aligned_m=float(res.mean()),
        median_aligned_m=float(np.median(res)),
        worst_aligned_m=float(res.max()),
        path_len_est_m=_path_length(est),
        path_len_gt_m=_path_length(gt),
        path_len_ratio=(_path_length(est) / _path_length(gt)) if _path_length(gt) else float("nan"),
        align=align,
    )
