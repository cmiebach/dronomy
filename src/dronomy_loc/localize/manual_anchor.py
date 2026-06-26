"""Manual anchoring: place a VO trajectory absolutely from a few human-marked
control points, when automatic satellite locks are too sparse to anchor it.

This is the deployable side of Adrian's hybrid (absolute fixes + VO between): an
operator marks the drone's position on the map for a handful of keyframes (or
recognises a landmark), and the relative VO track is fitted onto those points.
Frame-to-frame matching is easy everywhere (same camera, tiny baseline), so VO
gives a faithful *relative* path; its global scale, rotation and origin are
unknown. A best-fit **similarity** transform to >=2 control points recovers all
three at once and lays the whole track on the map.

GROUND-TRUTH RULE preserved: the control points are an *operator input*, not
telemetry/GPS auto-read by the system. (In evaluation we may simulate them from
GT to study how few anchors suffice — that is an experiment, not the runtime path.)

Pipeline:
  links  = pairwise_homographies(frames, matcher)   # odometry.py (reused)
  track  = anchor_trajectory(links, [ManualAnchor(f, lat, lon), ...])
  -> {frame_idx: (lat, lon)} for every frame reachable from the first.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..reference.geo import meters_per_degree_lat
from .odometry import PairwiseLink


@dataclass
class ManualAnchor:
    """An operator-supplied absolute position for one frame (the map click)."""
    frame_idx: int
    lat: float
    lon: float


def _decompose(H: np.ndarray) -> tuple[float, float, float]:
    """(theta, tx, ty) from a 3x3 homography: rotation of the linear part and the
    translation column (normalised by H[2,2]). Exact for an affine inter-frame
    motion and a good proxy for the small-baseline homographies VO produces."""
    H = np.asarray(H, dtype=np.float64)
    H = H / H[2, 2]
    return math.atan2(H[1, 0], H[0, 0]), float(H[0, 2]), float(H[1, 2])


def integrate_relative_track(links: list[PairwiseLink]) -> dict[int, tuple[float, float]]:
    """Integrate consecutive-frame homographies into a relative (x, y) track in
    camera units, starting the first frame at the origin. Each step's translation
    is rotated by the running heading so yaw is respected. Stops at the first
    tracking break (H is None) -- frames past an un-chainable gap are omitted."""
    pts: dict[int, tuple[float, float]] = {}
    if not links:
        return pts
    heading = x = y = 0.0
    pts[links[0].idx_from] = (0.0, 0.0)
    for lk in links:
        if lk.H is None:
            break
        dth, tx, ty = _decompose(lk.H)
        c, s = math.cos(heading), math.sin(heading)
        x += c * tx - s * ty
        y += s * tx + c * ty
        heading += dth
        pts[lk.idx_to] = (x, y)
    return pts


def fit_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Least-squares similarity (rotation + uniform scale + translation) mapping
    src -> dst (Umeyama 1991). Returns (R 2x2, s, t 2,). Unlike the rigid
    `trajectory.align_se2`, scale is recovered -- essential because VO units are
    arbitrary. Reflection is suppressed so R is a proper rotation."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = (dc.T @ sc) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    var = (sc ** 2).sum() / len(src)
    s = float((D * np.diag(S)).sum() / var) if var > 1e-12 else 1.0
    t = mu_d - s * (R @ mu_s)
    return R, s, t


def anchor_trajectory(
    links: list[PairwiseLink],
    anchors: list[ManualAnchor],
) -> dict[int, tuple[float, float]]:
    """Place the VO track absolutely from manual control points.

    Returns {frame_idx: (lat, lon)} for every frame on the integrated track.
    Needs >= 2 anchors (a similarity transform has 4 DOF; one point fixes only
    the origin). Anchors must reference frames that are on the track.
    """
    rel = integrate_relative_track(links)
    usable = [a for a in anchors if a.frame_idx in rel]
    if len(usable) < 2:
        raise ValueError(
            f"manual anchoring needs >= 2 control points on the track; got "
            f"{len(usable)} (of {len(anchors)} supplied). One point cannot fix "
            "the VO track's unknown scale and rotation.")

    ref_lat = float(np.mean([a.lat for a in usable]))
    ref_lon = float(np.mean([a.lon for a in usable]))
    m_lat, m_lon = meters_per_degree_lat(ref_lat)

    # Control points: relative-track coords (src) -> local east/north metres (dst).
    src = np.array([rel[a.frame_idx] for a in usable], dtype=float)
    dst = np.array([[(a.lon - ref_lon) * m_lon, (a.lat - ref_lat) * m_lat]
                    for a in usable], dtype=float)
    R, s, t = fit_similarity(src, dst)

    out: dict[int, tuple[float, float]] = {}
    for fidx, (x, y) in rel.items():
        east, north = s * (R @ np.array([x, y])) + t
        out[fidx] = (ref_lat + north / m_lat, ref_lon + east / m_lon)
    return out
