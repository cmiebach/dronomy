"""Visual-odometry chaining tests on a synthetic flight — fully offline.

A textured mercator 'world' GeoImage doubles as the reference; 'drone frames'
are axis-aligned integer-pixel crops of it, so the anchor's absolute homography
is known EXACTLY by construction (pure scale + translation) and every chained
pose can be scored in meters against the true crop centres.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronomy_loc.data.telemetry import GPSFix  # noqa: E402
from dronomy_loc.localize.odometry import (  # noqa: E402
    ChainResult, PairwiseLink, anchor_from, chain_poses, drift_curve,
    pairwise_homographies,
)
from dronomy_loc.matching.classical import ClassicalMatcher  # noqa: E402
from dronomy_loc.reference.geo import (  # noqa: E402
    GeoImage, haversine_m, mercator_bbox_around,
)

LAT, LON = 43.521955, -5.624290  # the Asturias coarse prior
WORLD_PX, WORLD_SPAN = 3072, 600.0
CROP, SIZE = 410, 512            # 410 world px ~ 80 m footprint -> 512 px frames
STEP = 77                        # ~15 m east per frame, in world px
X0, Y0 = 1100, 1300              # first crop's top-left (world px)
N_FRAMES = 8

# Cap SIFT features: full-noise images yield thousands, and 7 brute-force
# matches per flight would dominate the test budget.
_CFG = SimpleNamespace(matching=SimpleNamespace(
    classical=SimpleNamespace(detector="SIFT", max_features=1500, ratio_test=0.75),
    ransac=SimpleNamespace(reproj_threshold_px=5.0, confidence=0.999, min_inliers=12),
))


# ── synthetic world + flight ──────────────────────────────────────────
def make_world() -> GeoImage:
    bbox = mercator_bbox_around(LON, LAT, WORLD_SPAN)
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, (WORLD_PX, WORLD_PX, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # Bright shapes give SIFT strong corners/blobs on top of the smoothed noise.
    for _ in range(200):
        x, y = (int(v) for v in rng.integers(0, WORLD_PX - 80, 2))
        w, h = (int(v) for v in rng.integers(15, 70, 2))
        color = tuple(int(c) for c in rng.integers(160, 256, 3))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        else:
            cv2.circle(img, (x + 40, y + 40), w // 2 + 5, color, -1)
    return GeoImage(image=img, bbox=bbox)


def crop_rect(k: int) -> tuple[int, int]:
    return X0 + k * STEP, Y0


def make_flight(world: GeoImage, rot_step_deg: float = 0.0):
    """(frames [(idx, bgr)], gt [(lat, lon)]). Frame k is the crop at
    crop_rect(k) resized to SIZE, optionally rotated about its centre by
    k*rot_step_deg (centre stays fixed, so the GT centre is unchanged)."""
    frames, gt = [], []
    for k in range(N_FRAMES):
        x0, y0 = crop_rect(k)
        img = cv2.resize(world.image[y0:y0 + CROP, x0:x0 + CROP], (SIZE, SIZE),
                         interpolation=cv2.INTER_AREA)
        if rot_step_deg:
            M = cv2.getRotationMatrix2D((SIZE / 2.0, SIZE / 2.0), k * rot_step_deg, 1.0)
            img = cv2.warpAffine(img, M, (SIZE, SIZE), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
        frames.append((k, img))
        lon, lat = world.pixel_to_lonlat(x0 + CROP / 2.0, y0 + CROP / 2.0)
        gt.append((lat, lon))
    return frames, gt


def exact_anchor_h(k: int) -> np.ndarray:
    """Drone px -> world px for the UNROTATED frame k: scale + translation."""
    x0, y0 = crop_rect(k)
    s = CROP / SIZE
    return np.array([[s, 0.0, x0], [0.0, s, y0], [0.0, 0.0, 1.0]])


def make_track(gt: list[tuple[float, float]]) -> list[GPSFix]:
    return [GPSFix(frame=k, t_s=k / 29.97, lat=la, lon=lo, alt_m=50.0)
            for k, (la, lo) in enumerate(gt)]


@pytest.fixture(scope="module")
def world() -> GeoImage:
    return make_world()


@pytest.fixture(scope="module")
def flight(world):
    return make_flight(world)


@pytest.fixture(scope="module")
def links(flight):
    cv2.setRNGSeed(42)  # findHomography RANSAC uses cv2's global RNG
    return pairwise_homographies(flight[0], ClassicalMatcher(_CFG))


# ── (1) consecutive pairs all link, with strong inlier support ────────
def test_pairwise_all_links_ok(links):
    assert len(links) == N_FRAMES - 1
    for ln in links:
        assert ln.idx_to == ln.idx_from + 1
        assert ln.H is not None
        assert ln.n_inliers >= 30


# ── (2) one exact anchor at frame 0 recovers the whole flight ─────────
def test_chain_recovers_centers(world, flight, links):
    _, gt = flight
    chain = chain_poses(links, [anchor_from(0, exact_anchor_h(0), world)], (SIZE, SIZE))
    assert isinstance(chain, ChainResult)
    assert sorted(chain.poses) == list(range(N_FRAMES))
    for k, (glat, glon) in enumerate(gt):
        pose = chain.poses[k]
        assert haversine_m(glat, glon, pose.lat, pose.lon) < 3.0
        assert chain.hops[k] == k                  # hops grow monotonically
        assert chain.anchor_frame[k] == 0
        assert pose.frame_index == k


# ── (3) drift grows with hops but stays bounded on clean data ─────────
def test_drift_grows_with_hops(world):
    cv2.setRNGSeed(42)
    frames, gt = make_flight(world, rot_step_deg=3.0)   # 3 deg/frame cumulative
    links = pairwise_homographies(frames, ClassicalMatcher(_CFG))
    assert all(ln.H is not None for ln in links)
    chain = chain_poses(links, [anchor_from(0, exact_anchor_h(0), world)], (SIZE, SIZE))
    rows = drift_curve(chain, make_track(gt))
    err = {r["hops_from_anchor"]: r["err_m"] for r in rows}
    assert err[7] > err[1]                         # error accumulates outward
    assert all(e < 10.0 for e in err.values())     # but stays metric-bounded
    assert all(r["anchor_frame"] == 0 for r in rows)


# ── (4) a tracking break splits the chain; a second anchor heals it ───
def test_break_and_second_anchor(world, flight):
    cv2.setRNGSeed(42)
    frames, gt = flight
    broken = list(frames)
    broken[4] = (4, np.full((SIZE, SIZE, 3), 128, np.uint8))  # featureless frame
    links = pairwise_homographies(broken, ClassicalMatcher(_CFG))
    assert links[3].H is None and links[4].H is None          # both sides break

    a0 = anchor_from(0, exact_anchor_h(0), world)
    chain = chain_poses(links, [a0], (SIZE, SIZE))
    assert sorted(chain.poses) == [0, 1, 2, 3]     # beyond the break: omitted

    a6 = anchor_from(6, exact_anchor_h(6), world)
    chain2 = chain_poses(links, [a0, a6], (SIZE, SIZE))
    assert sorted(chain2.poses) == [0, 1, 2, 3, 5, 6, 7]      # gray frame 4 still out
    assert chain2.hops[5] == 1 and chain2.hops[6] == 0 and chain2.hops[7] == 1
    for k in (5, 6, 7):
        assert chain2.anchor_frame[k] == 6         # chained to the NEAR anchor
        glat, glon = gt[k]
        assert haversine_m(glat, glon, chain2.poses[k].lat, chain2.poses[k].lon) < 3.0


# ── (5) composition stays normalized over the full chain ──────────────
def test_chain_h_normalized(world, flight, links):
    chain = chain_poses(links, [anchor_from(0, exact_anchor_h(0), world)], (SIZE, SIZE))
    assert len(chain.H_to_ref) == N_FRAMES
    for H in chain.H_to_ref.values():
        assert np.all(np.isfinite(H))
        assert abs(H[2, 2] - 1.0) < 1e-6
        assert np.isfinite(np.linalg.det(H))


# ── (6) pure-math chain: exact links, two anchors, tie-breaking ───────
def test_exact_links_and_tie_break(world):
    # Hand-built exact links: H_{k->k+1} = inv(A_{k+1}) @ A_k (no matching).
    links = []
    for k in range(4):
        H = np.linalg.inv(exact_anchor_h(k + 1)) @ exact_anchor_h(k)
        links.append(PairwiseLink(k, k + 1, H, 99))
    anchors = [anchor_from(0, exact_anchor_h(0), world),
               anchor_from(4, exact_anchor_h(4), world)]
    chain = chain_poses(links, anchors, (SIZE, SIZE))
    assert chain.anchor_frame[1] == 0 and chain.anchor_frame[3] == 4
    assert chain.anchor_frame[2] == 0              # 2-hop tie -> lower frame_idx
    assert chain.hops == {0: 0, 1: 1, 2: 2, 3: 1, 4: 0}
    for k in range(5):
        x0, y0 = crop_rect(k)
        glon, glat = world.pixel_to_lonlat(x0 + CROP / 2.0, y0 + CROP / 2.0)
        pose = chain.poses[k]
        assert haversine_m(glat, glon, pose.lat, pose.lon) < 0.01  # exact chain
