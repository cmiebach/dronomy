"""Grid × scale search tests against a synthetic mercator 'world' — fully offline.

`fetch_tile` crops a big in-memory `GeoImage` through its own mercator<->pixel
mapping, so every tile is exactly georeferenced by construction and the search
can be checked against a known ground-truth point to sub-tile accuracy.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronomy_loc.localize.search import (  # noqa: E402
    TileCache, grid_centers, refine_localize, search_localize,
)
from dronomy_loc.matching.base import Matcher, MatchResult  # noqa: E402
from dronomy_loc.matching.classical import ClassicalMatcher  # noqa: E402
from dronomy_loc.reference.geo import (  # noqa: E402
    GeoImage, haversine_m, lonlat_to_mercator, mercator_bbox_around, mercator_to_lonlat,
)

LAT, LON = 43.521955, -5.624290  # the Asturias coarse prior
WORLD_PX, WORLD_SPAN = 3072, 600.0


# ── synthetic world + injected fetch_tile ─────────────────────────────
def make_world(blank: bool = False) -> GeoImage:
    bbox = mercator_bbox_around(LON, LAT, WORLD_SPAN)
    if blank:
        return GeoImage(image=np.full((WORLD_PX, WORLD_PX, 3), 128, np.uint8), bbox=bbox)
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


def make_fetch(world: GeoImage):
    """fetch_tile(lat, lon, span_m, pixels): crop the world via its mercator
    mapping (clamped to bounds) and resize — pure numpy/cv2, no network."""
    def fetch(lat: float, lon: float, span_m: float, pixels: int) -> GeoImage:
        minx, miny, maxx, maxy = mercator_bbox_around(lon, lat, span_m)
        x0, y0 = world.mercator_to_pixel(minx, maxy)   # top-left (row 0 == maxy)
        x1, y1 = world.mercator_to_pixel(maxx, miny)
        x0, y0 = max(0, round(x0)), max(0, round(y0))
        x1, y1 = min(world.width, round(x1)), min(world.height, round(y1))
        tile = cv2.resize(world.image[y0:y1, x0:x1], (pixels, pixels),
                          interpolation=cv2.INTER_AREA)
        # Re-derive the bbox from the ACTUAL pixel rect so clamping stays exact.
        gx0, gy0 = world.pixel_to_mercator(x0, y0)
        gx1, gy1 = world.pixel_to_mercator(x1, y1)
        return GeoImage(image=tile, bbox=(gx0, gy1, gx1, gy0))
    return fetch


def gt_point() -> tuple[float, float]:
    """Ground truth ~40 m (mercator) NE of the prior — off-grid on purpose."""
    cx, cy = lonlat_to_mercator(LON, LAT)
    lon, lat = mercator_to_lonlat(cx + 30.0, cy + 25.0)
    return lat, lon


def make_frame(world: GeoImage, lat: float, lon: float,
               span_m: float = 80.0, size: int = 512, rot_deg: float = 25.0):
    """'Drone frame': a north-up crop of the world at (lat, lon), rotated about
    its centre (centre stays fixed, so its ground point is still (lat, lon))."""
    crop = make_fetch(world)(lat, lon, span_m, size).image
    M = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), rot_deg, 1.0)
    return cv2.warpAffine(crop, M, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


@pytest.fixture(scope="module")
def world() -> GeoImage:
    return make_world()


# ── (1) grid geometry ─────────────────────────────────────────────────
def test_grid_centers_square_grid():
    centers = grid_centers(LAT, LON, 120.0, 60.0)
    assert len(centers) == 25                      # offsets -120..120 step 60, squared
    assert (LAT, LON) in centers                   # prior included exactly
    cx, cy = lonlat_to_mercator(LON, LAT)
    for lat, lon in centers:
        x, y = lonlat_to_mercator(lon, lat)
        assert abs(x - cx) <= 120.0 + 1e-3
        assert abs(y - cy) <= 120.0 + 1e-3


def test_grid_centers_degenerate():
    assert grid_centers(LAT, LON, 0.0, 60.0) == [(LAT, LON)]
    assert grid_centers(LAT, LON, 50.0, 100.0) == [(LAT, LON)]   # step >= 2*radius


# ── (2) full search recovers a known pose ─────────────────────────────
def test_search_recovers_ground_truth(world):
    cv2.setRNGSeed(42)  # findHomography RANSAC uses cv2's global RNG
    gt_lat, gt_lon = gt_point()
    frame = make_frame(world, gt_lat, gt_lon, span_m=80.0, size=512, rot_deg=25.0)
    res = search_localize(
        frame, LAT, LON, ClassicalMatcher(), make_fetch(world),
        search_radius_m=60.0, grid_step_m=60.0, scales_m=(60.0, 80.0, 100.0),
        pixels=640,
    )
    assert res.locked
    pose = res.best.pose
    assert haversine_m(gt_lat, gt_lon, pose.lat, pose.lon) < 5.0
    # Yaw convention: getRotationMatrix2D(+25) turns image CONTENT counter-
    # clockwise on screen (y down), so the frame's up-vector lands 25 deg east
    # of north on the north-up tile; pipeline's atan2(dx, -dy) reads that as
    # +25, not 335 (verified empirically against pose_from_homography).
    assert abs((pose.yaw_deg - 25.0 + 180.0) % 360.0 - 180.0) < 3.0


# ── (3) featureless world must not lock ───────────────────────────────
def test_blank_world_no_lock():
    blank = make_world(blank=True)
    frame = np.full((512, 512, 3), 128, np.uint8)
    res = search_localize(
        frame, LAT, LON, ClassicalMatcher(), make_fetch(blank),
        search_radius_m=60.0, grid_step_m=60.0, scales_m=(60.0, 80.0),
    )
    assert not res.locked
    assert res.best is None or res.best.n_inliers < 20


# ── (4) cache: a repeated search must not refetch ─────────────────────
def test_tile_cache_no_refetch_on_second_search():
    blank = make_world(blank=True)
    base, calls = make_fetch(blank), []

    def counting(lat, lon, span_m, pixels):
        calls.append((lat, lon, span_m, pixels))
        return base(lat, lon, span_m, pixels)

    cache = TileCache(counting)
    frame = np.full((256, 256, 3), 128, np.uint8)
    kwargs = dict(search_radius_m=60.0, grid_step_m=60.0, scales_m=(60.0, 80.0))
    search_localize(frame, LAT, LON, ClassicalMatcher(), cache, **kwargs)
    assert len(calls) == 9 * 2                     # 3x3 centres x 2 spans
    search_localize(frame, LAT, LON, ClassicalMatcher(), cache, **kwargs)
    assert len(calls) == 9 * 2                     # second run: all cache hits


# ── (5) one bad tile fetch must not kill the search ───────────────────
class _StubMatcher(Matcher):
    """Always 'locks' with 24 identity-homography inliers — isolates the
    search's failure handling from real feature matching."""
    def match(self, drone_bgr, ref_rgb) -> MatchResult:
        pts = np.tile(np.float32([[10, 10], [100, 10], [10, 100], [100, 100]]), (6, 1))
        return MatchResult(pts, pts, np.eye(3), np.ones(len(pts), bool), len(pts))


# ── (6) relative-margin lock gate (the RoMA / dense-matcher gate) ──────
class _DenseStub(Matcher):
    """Mimics a DENSE matcher: returns an identity-homography 'match' with an
    inlier count baked into the ref tile's pixel [0,0,0]. Dense matchers score
    high inliers on EVERY tile, so the number — not its mere presence — is what
    a real lock has to defend via the margin gate."""
    def match(self, drone_bgr, ref_rgb) -> MatchResult:
        n = max(int(ref_rgb[0, 0, 0]), 1)
        pts = np.zeros((n, 2), np.float32)
        return MatchResult(pts, pts, np.eye(3), np.ones(n, bool), n)


def make_scored_fetch(score_fn):
    """Tile geometry from a blank world (so an identity-H pose lands on the tile
    centre), with `score_fn(lat, lon)` baked into pixel [0,0,0] as the inlier
    count `_DenseStub` will report for that tile."""
    base = make_fetch(make_world(blank=True))

    def fetch(lat, lon, span_m, pixels):
        tile = base(lat, lon, span_m, pixels)
        img = tile.image.copy()
        img[0, 0, 0] = int(np.clip(score_fn(lat, lon), 0, 255))
        return GeoImage(image=img, bbox=tile.bbox)
    return fetch


def _cell_score(lat, lon, *, peak, neighbor, distant):
    """Score by mercator distance from the prior: the peak cell, its near ring,
    or a far-away alternative hypothesis."""
    cx, cy = lonlat_to_mercator(LON, LAT)
    x, y = lonlat_to_mercator(lon, lat)
    d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
    if d < 1.0:
        return peak
    if d < 90.0:
        return neighbor
    return distant


# A 640x640 frame + pixels=640 => identity-H pose maps the frame centre onto the
# tile centre, so each candidate's estimate sits exactly on its grid centre and
# separations equal the grid geometry.
_FRAME = np.zeros((640, 640, 3), np.uint8)
_GRID = dict(search_radius_m=120.0, grid_step_m=60.0, scales_m=(80.0,), pixels=640)


def test_margin_gate_rejects_confident_wrong_lock():
    # Peak only 10% above distant rivals: a dense matcher would lock confidently,
    # but the margin says the hypothesis is not separable -> must NOT lock.
    fetch = make_scored_fetch(lambda la, lo: _cell_score(
        la, lo, peak=110, neighbor=100, distant=100))
    res = search_localize(_FRAME, LAT, LON, _DenseStub(), fetch,
                          lock_margin_ratio=1.3, margin_separation_m=90.0, **_GRID)
    assert res.best.n_inliers == 110            # absolute gate (>=20) passes
    assert res.runner_up is not None and res.runner_up.n_inliers == 100
    assert abs(res.margin_ratio - 1.1) < 1e-6
    assert not res.locked                       # margin gate vetoes it
    # Same search WITHOUT the margin gate would have locked (the old behaviour).
    inert = search_localize(_FRAME, LAT, LON, _DenseStub(), fetch,
                            lock_margin_ratio=1.0, margin_separation_m=90.0, **_GRID)
    assert inert.locked


def test_margin_gate_locks_dominant_peak():
    fetch = make_scored_fetch(lambda la, lo: _cell_score(
        la, lo, peak=200, neighbor=100, distant=100))
    res = search_localize(_FRAME, LAT, LON, _DenseStub(), fetch,
                          lock_margin_ratio=1.3, margin_separation_m=90.0, **_GRID)
    assert res.locked
    assert res.best.n_inliers == 200
    assert abs(res.margin_ratio - 2.0) < 1e-6
    # The winning estimate sits on the prior (the peak cell).
    assert haversine_m(LAT, LON, res.best.pose.lat, res.best.pose.lon) < 5.0


def test_runner_up_excludes_same_peak_neighbours():
    # A single dominant peak (200) with an elevated shoulder ring (170) and far
    # cells at 100. With a real separation the shoulder is not a rival, so the
    # peak beats the distant 100 by 2.0 and locks. Drop the separation to 0 and
    # the 170 shoulder becomes the rival (200/170 = 1.18 < 1.3) -> no lock:
    # proves the separation is what stops a peak shadowing itself.
    fetch = make_scored_fetch(lambda la, lo: _cell_score(
        la, lo, peak=200, neighbor=170, distant=100))
    locked = search_localize(_FRAME, LAT, LON, _DenseStub(), fetch,
                             lock_margin_ratio=1.3, margin_separation_m=90.0, **_GRID)
    assert locked.locked
    assert abs(locked.margin_ratio - 2.0) < 1e-6   # rival is a distant 100 cell
    leaky = search_localize(_FRAME, LAT, LON, _DenseStub(), fetch,
                            lock_margin_ratio=1.3, margin_separation_m=0.0, **_GRID)
    assert not leaky.locked
    assert abs(leaky.margin_ratio - 200 / 170) < 1e-6   # rival is the 170 shoulder


def test_margin_gate_inert_by_default():
    # Default ratio 1.0 keeps the sparse-matcher behaviour: a uniformly-confident
    # field still locks (absolute gate only), margin reported but not enforced.
    fetch = make_scored_fetch(lambda la, lo: 100)
    res = search_localize(_FRAME, LAT, LON, _DenseStub(), fetch, **_GRID)
    assert res.locked
    assert abs(res.margin_ratio - 1.0) < 1e-6


# ── (7) coarse-to-fine refinement tightens the estimate ───────────────
def _scored_by_distance(true_dx, true_dy, *, peak=250.0, decay=2.0, floor=30.0):
    """Inlier score peaking at the point `(true_dx, true_dy)` metres (mercator)
    from the prior, decaying linearly with distance: a deterministic stand-in for
    'this tile centre is close to the true ground point'. Baked into pixel
    [0,0,0] by `make_scored_fetch` and read back by `_DenseStub`."""
    cx, cy = lonlat_to_mercator(LON, LAT)
    tx, ty = cx + true_dx, cy + true_dy

    def fn(lat, lon):
        x, y = lonlat_to_mercator(lon, lat)
        d = ((x - tx) ** 2 + (y - ty) ** 2) ** 0.5
        return max(peak - decay * d, floor)
    return fn


def _true_lonlat(true_dx, true_dy):
    cx, cy = lonlat_to_mercator(LON, LAT)
    lon, lat = mercator_to_lonlat(cx + true_dx, cy + true_dy)
    return lat, lon


def test_refine_tightens_offgrid_estimate():
    # Truth sits at (22, 18) m — between coarse 60 m grid cells, so the coarse
    # winner is the prior cell ~28 m off. A fine 20 m grid around it lands a cell
    # ~3 m from truth: refinement must cut the error without losing the lock.
    true_lat, true_lon = _true_lonlat(22.0, 18.0)
    fetch = make_scored_fetch(_scored_by_distance(22.0, 18.0))
    coarse = search_localize(_FRAME, LAT, LON, _DenseStub(), fetch, **_GRID)
    assert coarse.locked
    coarse_err = haversine_m(true_lat, true_lon,
                             coarse.best.pose.lat, coarse.best.pose.lon)

    refined = refine_localize(_FRAME, coarse, _DenseStub(), fetch,
                              refine_radius_m=40.0, refine_step_m=20.0,
                              scale_factors=(1.0,))
    fine_err = haversine_m(true_lat, true_lon,
                           refined.best.pose.lat, refined.best.pose.lon)
    assert refined.locked
    assert coarse_err > 20.0                 # coarse grid cannot reach the truth
    assert fine_err < 5.0                     # fine grid does
    assert fine_err < coarse_err
    assert refined.best.n_inliers >= coarse.best.n_inliers   # never regress
    # The lock decision/margin are carried over verbatim, not recomputed.
    assert refined.margin_ratio == coarse.margin_ratio
    assert refined.runner_up is coarse.runner_up
    # Union of both passes is kept for inspection.
    assert len(refined.candidates) > len(coarse.candidates)


def test_refine_skips_unlocked_coarse_result():
    # A uniformly weak field (10 inliers everywhere) never locks; refining it
    # would just relocate noise, so the coarse result is returned untouched.
    fetch = make_scored_fetch(lambda la, lo: 10)
    coarse = search_localize(_FRAME, LAT, LON, _DenseStub(), fetch, **_GRID)
    assert not coarse.locked
    refined = refine_localize(_FRAME, coarse, _DenseStub(), fetch)
    assert refined is coarse                  # same object, no fine pass run


def test_refine_preserves_real_matcher_lock(world):
    # End-to-end with the real SIFT pipeline (not the stub): refinement must keep
    # the lock and never score below the coarse best (the max() guard), even
    # though RANSAC's RNG state differs on the re-matched tiles.
    cv2.setRNGSeed(42)
    gt_lat, gt_lon = gt_point()
    frame = make_frame(world, gt_lat, gt_lon, span_m=80.0, size=512, rot_deg=25.0)
    fetch = TileCache(make_fetch(world))
    kw = dict(search_radius_m=60.0, grid_step_m=60.0,
              scales_m=(60.0, 80.0, 100.0), pixels=640)
    coarse = search_localize(frame, LAT, LON, ClassicalMatcher(), fetch, **kw)
    assert coarse.locked
    refined = refine_localize(frame, coarse, ClassicalMatcher(), fetch)
    assert refined.locked
    assert refined.best.n_inliers >= coarse.best.n_inliers
    coarse_err = haversine_m(gt_lat, gt_lon,
                             coarse.best.pose.lat, coarse.best.pose.lon)
    fine_err = haversine_m(gt_lat, gt_lon,
                           refined.best.pose.lat, refined.best.pose.lon)
    assert fine_err <= coarse_err + 2.0       # never meaningfully regress


def test_one_bad_tile_fetch_does_not_kill_search():
    blank = make_world(blank=True)
    base = make_fetch(blank)
    bad_lat, bad_lon = grid_centers(LAT, LON, 60.0, 60.0)[3]

    def flaky(lat, lon, span_m, pixels):
        if (round(lat, 7), round(lon, 7)) == (round(bad_lat, 7), round(bad_lon, 7)):
            raise RuntimeError("tile service hiccup")
        return base(lat, lon, span_m, pixels)

    frame = np.zeros((64, 64, 3), np.uint8)
    res = search_localize(
        frame, LAT, LON, _StubMatcher(), flaky,
        search_radius_m=60.0, grid_step_m=60.0, scales_m=(60.0, 80.0),
    )
    assert len(res.candidates) == 9 * 2            # failed cells still recorded
    failed = [c for c in res.candidates if c.pose is None]
    assert len(failed) == 2                        # the bad centre x both spans
    assert all(c.n_inliers == 0 for c in failed)
    assert res.locked and res.best.n_inliers == 24  # other candidates intact
