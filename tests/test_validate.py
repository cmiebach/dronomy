"""Validation-harness tests — fully offline, synthetic world + synthetic video.

Reuses the synthetic mercator 'world' pattern from test_search.py (tiles cropped
from one in-memory GeoImage are exactly georeferenced by construction) and the
cv2.VideoWriter clip pattern from test_ingest.py, so the whole harness — frame
grabbing, world-crop fetch, scoring, CSV round-trip — is checked against known
ground truth with no network and no real footage.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronomy_loc.data.telemetry import GPSFix  # noqa: E402
from dronomy_loc.localize.search import TileCache  # noqa: E402
from dronomy_loc.localize.validate import (  # noqa: E402
    grab_frames, make_world_fetch, parse_frames_spec, read_validation_csv,
    validate_frames, write_validation_csv,
)
from dronomy_loc.matching.classical import ClassicalMatcher  # noqa: E402
from dronomy_loc.reference.geo import (  # noqa: E402
    GeoImage, haversine_m, lonlat_to_mercator, mercator_bbox_around, mercator_to_lonlat,
)

LAT, LON = 43.521955, -5.624290  # the Asturias coarse prior
WORLD_PX, WORLD_SPAN = 3072, 600.0


# ── synthetic world + frames (same pattern as test_search.py) ─────────
def make_world() -> GeoImage:
    bbox = mercator_bbox_around(LON, LAT, WORLD_SPAN)
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, (WORLD_PX, WORLD_PX, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    for _ in range(200):
        x, y = (int(v) for v in rng.integers(0, WORLD_PX - 80, 2))
        w, h = (int(v) for v in rng.integers(15, 70, 2))
        color = tuple(int(c) for c in rng.integers(160, 256, 3))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        else:
            cv2.circle(img, (x + 40, y + 40), w // 2 + 5, color, -1)
    return GeoImage(image=img, bbox=bbox)


def gt_point() -> tuple[float, float]:
    cx, cy = lonlat_to_mercator(LON, LAT)
    lon, lat = mercator_to_lonlat(cx + 30.0, cy + 25.0)
    return lat, lon


def make_frame(world: GeoImage, lat: float, lon: float,
               span_m: float = 80.0, size: int = 512, rot_deg: float = 25.0):
    """'Drone frame': north-up crop rotated about its centre (centre fixed,
    so its ground point is still (lat, lon))."""
    crop = make_world_fetch(world)(lat, lon, span_m, size).image
    M = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), rot_deg, 1.0)
    return cv2.warpAffine(crop, M, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def make_video(path: Path, n_frames: int, fps: int = 10, size=(64, 48)) -> Path:
    """Tiny deterministic clip (pattern from test_ingest.py)."""
    w, h = size
    for fourcc, suffix in (("mp4v", ".mp4"), ("MJPG", ".avi")):
        p = path.with_suffix(suffix)
        vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        if not vw.isOpened():
            continue
        rng = np.random.default_rng(7)
        for i in range(n_frames):
            img = rng.integers(0, 40, (h, w, 3), dtype=np.uint8)
            x = (i * 3) % max(1, w - 12)
            cv2.rectangle(img, (x, 8), (x + 10, 28), (255, 255, 255), -1)
            vw.write(img)
        vw.release()
        cap = cv2.VideoCapture(str(p))
        opened = cap.isOpened()
        cap.release()
        if opened:
            return p
    pytest.skip("no usable cv2.VideoWriter backend")


@pytest.fixture(scope="module")
def world() -> GeoImage:
    return make_world()


# ── (1) parse_frames_spec ─────────────────────────────────────────────
def test_parse_frames_spec_explicit_list():
    assert parse_frames_spec("342,3083,6510", 6853) == [342, 3083, 6510]
    assert parse_frames_spec(" 6510, 342 ,342 ", 6853) == [342, 6510]  # sorted unique


def test_parse_frames_spec_count_spread():
    idxs = parse_frames_spec("12", 6853)
    assert len(idxs) == 12
    assert idxs[0] == 0 and idxs[-1] == 6852           # endpoints included
    assert idxs == sorted(set(idxs))                   # sorted unique


def test_parse_frames_spec_single_int_string():
    assert parse_frames_spec("1", 100) == [0]
    assert parse_frames_spec("3", 101) == [0, 50, 100]
    assert parse_frames_spec("5", 3) == [0, 1, 2]      # count > n_total dedupes


def test_parse_frames_spec_junk_raises():
    for bad in ("abc", "", "12.5", "0", "-3", "1,zz,3"):
        with pytest.raises(ValueError):
            parse_frames_spec(bad, 100)
    with pytest.raises(ValueError):
        parse_frames_spec("10,9999", 100)              # out of range
    with pytest.raises(ValueError):
        parse_frames_spec("12", 0)                     # empty video


# ── (2) make_world_fetch geometry ─────────────────────────────────────
def test_make_world_fetch_bbox(world):
    fetch = make_world_fetch(world)
    tile = fetch(LAT, LON, 100.0, 256)
    assert tile.image.shape == (256, 256, 3)

    # Centre round-trips within 1 m (clamp rounding is <= 0.5 world px ~ 0.1 m).
    lon_c, lat_c = tile.pixel_to_lonlat(tile.width / 2.0, tile.height / 2.0)
    assert haversine_m(LAT, LON, lat_c, lon_c) < 1.0

    # Corner agrees with the requested bbox within 1 world-px equivalent.
    minx, miny, maxx, maxy = mercator_bbox_around(LON, LAT, 100.0)
    gx, gy = lonlat_to_mercator(*tile.pixel_to_lonlat(0.0, 0.0))
    mpp_x, mpp_y = world.meters_per_pixel
    assert abs(gx - minx) <= mpp_x and abs(gy - maxy) <= mpp_y


def test_make_world_fetch_clamps_at_world_edge(world):
    fetch = make_world_fetch(world)
    cx, cy = lonlat_to_mercator(LON, LAT)
    lon_e, lat_e = mercator_to_lonlat(cx + 280.0, cy)  # 280 m east, 20 m from edge
    tile = fetch(lat_e, lon_e, 100.0, 128)
    assert tile.image.shape == (128, 128, 3)           # raster still square
    minx, miny, maxx, maxy = tile.bbox
    wminx, wminy, wmaxx, wmaxy = world.bbox
    assert maxx == pytest.approx(wmaxx)                # pinned to the world edge
    assert 65.0 < (maxx - minx) < 75.0                 # clamped: ~70 m, not 100
    assert 99.0 < (maxy - miny) < 101.0                # untouched axis keeps 100 m

    # A request entirely outside the world must raise, not return garbage.
    lon_out, lat_out = mercator_to_lonlat(cx + 400.0, cy)
    with pytest.raises(ValueError):
        fetch(lat_out, lon_out, 100.0, 128)


# ── (3) end-to-end: two frames, known GT, CSV round-trip ──────────────
def test_validate_frames_end_to_end(world, tmp_path):
    cv2.setRNGSeed(42)  # findHomography RANSAC uses cv2's global RNG
    gt_lat, gt_lon = gt_point()
    frames = {
        100: make_frame(world, gt_lat, gt_lon, span_m=80.0, size=512, rot_deg=25.0),
        200: np.full((512, 512, 3), 128, np.uint8),    # featureless: cannot lock
    }
    track = [
        GPSFix(frame=100, t_s=100 / 29.97, lat=gt_lat, lon=gt_lon, alt_m=50.0),
        GPSFix(frame=200, t_s=200 / 29.97, lat=LAT, lon=LON, alt_m=50.0),
    ]
    fetch = TileCache(make_world_fetch(world))         # ONE cache for all frames
    seen = []
    summary = validate_frames(
        frames, track, LAT, LON, ClassicalMatcher(), fetch,
        search_radius_m=60.0, grid_step_m=60.0, scales_m=(60.0, 80.0, 100.0),
        pixels=640, on_row=seen.append,
    )

    assert summary.n == 2 and summary.n_locked == 1
    assert summary.lock_rate == pytest.approx(0.5)
    assert len(seen) == 2

    locked = next(r for r in summary.rows if r.locked)
    assert locked.frame == 100
    assert locked.err_m is not None and locked.err_m < 5.0
    assert locked.n_inliers >= 20 and locked.runtime_s > 0.0
    assert (locked.gt_lat, locked.gt_lon) == (gt_lat, gt_lon)

    unlocked = next(r for r in summary.rows if not r.locked)
    assert unlocked.frame == 200
    assert unlocked.err_m is None and unlocked.est_lat is None
    assert unlocked.n_inliers < 20

    # Stats over LOCKED frames only: with one lock, all three equal its error.
    assert summary.median_err_m == summary.mean_err_m == summary.worst_err_m
    assert summary.median_err_m == pytest.approx(locked.err_m)

    out = tmp_path / "sub" / "validation.csv"          # parent dir gets created
    write_validation_csv(summary, out)
    assert out.exists() and not out.with_suffix(".csv.tmp").exists()
    assert read_validation_csv(out) == summary.rows    # exact round-trip


def test_summary_none_when_nothing_locks(world):
    frames = {0: np.full((256, 256, 3), 128, np.uint8)}
    track = [GPSFix(frame=0, t_s=0.0, lat=LAT, lon=LON, alt_m=50.0)]
    summary = validate_frames(
        frames, track, LAT, LON, ClassicalMatcher(), make_world_fetch(world),
        search_radius_m=0.0, grid_step_m=60.0, scales_m=(80.0,),
    )
    assert summary.n == 1 and summary.n_locked == 0 and summary.lock_rate == 0.0
    assert summary.median_err_m is None
    assert summary.mean_err_m is None
    assert summary.worst_err_m is None


# ── (4) grab_frames: one sequential pass on a synthetic video ─────────
def test_grab_frames_returns_requested_indices(tmp_path):
    video = make_video(tmp_path / "vid", 20)
    got = grab_frames(video, [17, 3, 0, 3], resize_long_edge=None)
    assert sorted(got) == [0, 3, 17]
    for img in got.values():
        assert img.shape == (48, 64, 3)
    assert not np.array_equal(got[0], got[17])         # really different frames

    small = grab_frames(video, [0], resize_long_edge=32)
    assert small[0].shape == (24, 32, 3)


def test_grab_frames_raises_on_miss(tmp_path):
    video = make_video(tmp_path / "vid", 10)
    with pytest.raises(RuntimeError, match="could not grab"):
        grab_frames(video, [5, 25])                    # 25 is past the end
    with pytest.raises(ValueError):
        grab_frames(video, [-1, 2])
    with pytest.raises(FileNotFoundError):
        grab_frames(tmp_path / "nope.mp4", [0])
