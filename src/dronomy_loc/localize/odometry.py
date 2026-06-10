"""Visual-odometry dead-reckoning: extend absolute localization to frames that
cannot match the satellite map directly.

Only a small fraction (~6%) of this flight's frames lock against the reference
imagery; the rest are too oblique, too bland, or too seasonally changed. But the
ground is near-planar and the camera near-nadir at near-constant altitude
(~50 m), so CONSECUTIVE frames k-1, k relate by a plane-induced homography
H_{k-1->k} that classical matching estimates very reliably (huge overlap, same
sensor, same lighting). Given an 'anchor' frame a with a verified absolute
registration H_{a->ref} (drone px -> reference-tile px, straight from its
locked match), any frame k chains onto the map:

    H_{k->ref} = H_{a->ref} @ H_{k->a}

where H_{k->a} is the product of the pairwise links between k and a (links are
inverted when walking forward in time past the anchor). `pose_from_homography`
then yields (lat, lon, yaw) for frame k exactly as for a direct match.

Honesty about drift: every link contributes a small homography error, so the
chained pose degrades with hop count from the anchor — this is dead-reckoning,
not magic. Each additional anchor resets the error to its absolute accuracy, so
`chain_poses` assigns every frame to its NEAREST anchor (by hop count, stopping
at tracking breaks) and `drift_curve` reports error vs hops against the GPS
ground truth, keeping the drift measured rather than assumed.

Numeric constraint: homographies are scale-ambiguous, and products of many 3x3
matrices wander away from H[2,2] == 1 until values overflow or underflow. Every
composition here is renormalized by H[2,2] after every multiply.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.telemetry import GPSFix, gt_for_frame
from ..matching.base import Matcher
from ..reference.geo import GeoImage, haversine_m
from .pipeline import PoseEstimate, pose_from_homography


def _norm_h(H: np.ndarray) -> np.ndarray:
    """float64 copy divided by H[2,2] so repeated products stay tame."""
    H = np.asarray(H, dtype=np.float64)
    return H / H[2, 2]


@dataclass
class PairwiseLink:
    """Homography between CONSECUTIVE frames of a sweep. H is None on a
    tracking break (blur, dropout, featureless frame): chains must stop there."""
    idx_from: int
    idx_to: int
    H: np.ndarray | None           # 3x3, idx_from px -> idx_to px
    n_inliers: int = 0


def pairwise_homographies(
    frames: list[tuple[int, np.ndarray]],
    matcher: Matcher,
    *,
    min_inliers: int = 12,
) -> list[PairwiseLink]:
    """Match each consecutive pair of `frames` ([(frame_idx, image_bgr), ...],
    ascending frame_idx). A pair that fails to match — or clears fewer than
    `min_inliers` RANSAC inliers — yields H=None: an explicit break, never a
    guessed link, because one bad link silently corrupts every pose beyond it."""
    links: list[PairwiseLink] = []
    for (i0, img0), (i1, img1) in zip(frames, frames[1:]):
        try:
            mr = matcher.match(img0, img1)
        except Exception:
            links.append(PairwiseLink(i0, i1, None, 0))
            continue
        if mr.ok and mr.n_inliers >= min_inliers:
            links.append(PairwiseLink(i0, i1, _norm_h(mr.homography), mr.n_inliers))
        else:
            links.append(PairwiseLink(i0, i1, None, mr.n_inliers))
    return links


@dataclass
class Anchor:
    """An absolutely-registered frame: `H_to_ref` maps its pixels onto `ref`
    tile pixels. A locked `search_localize` provides both pieces directly —
    the best candidate's `MatchResult.homography` (drone px -> tile px, from
    `localize_frame`) and the tile fetched for the winning (centre, span)."""
    frame_idx: int
    H_to_ref: np.ndarray
    ref: GeoImage


def anchor_from(frame_idx: int, H: np.ndarray, ref_tile: GeoImage) -> Anchor:
    """Plain constructor that normalizes H. See `Anchor` for where the
    arguments come from in production."""
    return Anchor(frame_idx, _norm_h(H), ref_tile)


@dataclass
class ChainResult:
    """Chained poses plus the bookkeeping `drift_curve` needs. `n_inliers` on
    each pose is the WEAKEST link traversed (0 for the anchor frame itself):
    a chain is only as trustworthy as its worst pairwise match."""
    poses: dict[int, PoseEstimate]      # frame_idx -> chained pose
    hops: dict[int, int]                # frame_idx -> links traversed to anchor
    anchor_frame: dict[int, int]        # frame_idx -> anchor it chained to
    H_to_ref: dict[int, np.ndarray]     # frame_idx -> composed drone->ref-px H


def chain_poses(
    links: list[PairwiseLink],
    anchors: list[Anchor],
    frame_shapes: dict[int, tuple] | tuple,
) -> ChainResult:
    """Compose every reachable frame onto its NEAREST anchor and emit poses.

    The link list is a path graph; each anchor walks outward in both directions,
    stopping at breaks (H=None). Nearest anchor (fewest hops) wins; ties go to
    the lower anchor frame_idx. Frames unreachable from any anchor are omitted —
    no pose is better than a fabricated one. `frame_shapes` is a single (h, w)
    applied to all frames (the common case) or a per-frame dict."""
    nodes = ([links[0].idx_from] + [ln.idx_to for ln in links]) if links else []
    pos = {idx: i for i, idx in enumerate(nodes)}
    # frame -> (hops, anchor_idx, H_to_ref, ref, weakest_inliers)
    best: dict[int, tuple[int, int, np.ndarray, GeoImage, int]] = {}

    def offer(frame: int, hops: int, a: Anchor, H: np.ndarray, weakest: int) -> None:
        cur = best.get(frame)
        if cur is None or hops < cur[0] or (hops == cur[0] and a.frame_idx < cur[1]):
            best[frame] = (hops, a.frame_idx, H, a.ref, weakest)

    for a in sorted(anchors, key=lambda x: x.frame_idx):
        H0 = _norm_h(a.H_to_ref)
        offer(a.frame_idx, 0, a, H0, 0)
        p = pos.get(a.frame_idx)
        if p is None:
            continue
        # Backward in time: H_{k->ref} = H_{k+1->ref} @ H_{k->k+1}.
        H, weakest = H0, 10 ** 9
        for i in range(p - 1, -1, -1):
            ln = links[i]
            if ln.H is None:
                break
            H = _norm_h(H @ ln.H)
            weakest = min(weakest, ln.n_inliers)
            offer(nodes[i], p - i, a, H, weakest)
        # Forward in time: H_{k->ref} = H_{k-1->ref} @ inv(H_{k-1->k}).
        H, weakest = H0, 10 ** 9
        for i in range(p, len(links)):
            ln = links[i]
            if ln.H is None:
                break
            H = _norm_h(H @ np.linalg.inv(ln.H))
            weakest = min(weakest, ln.n_inliers)
            offer(nodes[i + 1], i + 1 - p, a, H, weakest)

    per_frame = frame_shapes if isinstance(frame_shapes, dict) else None
    poses: dict[int, PoseEstimate] = {}
    hops: dict[int, int] = {}
    anchor_of: dict[int, int] = {}
    chained_h: dict[int, np.ndarray] = {}
    for frame in sorted(best):
        n_hops, a_idx, H, ref, weakest = best[frame]
        shape = per_frame[frame] if per_frame is not None else frame_shapes
        pose = pose_from_homography(H, shape, ref)
        pose.frame_index = frame
        pose.n_inliers = 0 if n_hops == 0 else weakest
        poses[frame] = pose
        hops[frame] = n_hops
        anchor_of[frame] = a_idx
        chained_h[frame] = H
    return ChainResult(poses, hops, anchor_of, chained_h)


def drift_curve(chain: ChainResult, track: list[GPSFix]) -> list[dict]:
    """Per-frame error vs GPS ground truth, tagged with hop count and anchor —
    the 'how fast does dead-reckoning drift' deliverable, ready to plot as
    error-vs-hops. Ground truth is matched by nearest frame (`gt_for_frame`)."""
    rows: list[dict] = []
    for frame in sorted(chain.poses):
        pose = chain.poses[frame]
        gt = gt_for_frame(track, frame)
        rows.append({
            "frame": frame,
            "hops_from_anchor": chain.hops[frame],
            "anchor_frame": chain.anchor_frame[frame],
            "err_m": haversine_m(gt.lat, gt.lon, pose.lat, pose.lon),
            "est_lat": pose.lat, "est_lon": pose.lon,
            "gt_lat": gt.lat, "gt_lon": gt.lon,
        })
    return rows
