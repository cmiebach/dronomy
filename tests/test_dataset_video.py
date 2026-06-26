"""VideoDataset adapter tests — fully offline, deterministic.

A tiny synthetic clip (cv2.VideoWriter) plus a hand-written GPS CSV stand in for
the real flight. No network, no GPU, no real video. The cfg is a SimpleNamespace
mirroring the real config namespaces, and every path is ABSOLUTE so that
`config.resolve()` (which would otherwise join against the repo root) returns it
unchanged. We never CALL scenario.reference() — only assert it is callable — so
the test stays offline even though the adapter's fallback is a live provider.
"""
import csv
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronomy_loc.datasets.video import VideoDataset  # noqa: E402

N_FRAMES = 6
EVERY_N_SECONDS = 0.1   # fps=10 -> step 1 -> one sample per frame -> N_FRAMES samples


def make_video(path: Path, n_frames: int, fps: int = 10, size=(64, 48)) -> Path:
    """Tiny deterministic clip: moving rectangle + seeded noise so frames differ.
    Tries mp4v/.mp4 then MJPG/.avi; skips only if neither backend works."""
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


def write_gps_csv(path: Path, n_frames: int) -> None:
    """One fix per frame; lat/lon march linearly so nearest-frame lookup is
    distinguishable per frame."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "t_s", "lat", "lon", "alt_m"])
        for i in range(n_frames):
            w.writerow([i, i * 0.1, 43.5 + i * 0.001, -5.6 - i * 0.001, 50.0 + i])


def make_cfg(tmp_path: Path, video: Path, gps_csv: Path) -> types.SimpleNamespace:
    """Mirror the real nested namespaces. ABSOLUTE paths so resolve() is a no-op."""
    return types.SimpleNamespace(
        video=types.SimpleNamespace(
            path=str(video),
            rough_lat=43.521955,
            rough_lon=-5.624290,
            gps_track_csv=str(gps_csv),
        ),
        camera=types.SimpleNamespace(focal_px=3713.0, hfov_deg=84.0),
        frames=types.SimpleNamespace(every_n_seconds=EVERY_N_SECONDS,
                                     resize_long_edge=1920),
        reference=types.SimpleNamespace(
            provider="esri",
            out_dir=str(tmp_path / "no_such_reference_dir"),  # absent -> live-provider fallback
        ),
    )


def test_scenarios_single_scenario_with_samples(tmp_path):
    video = make_video(tmp_path / "vid", N_FRAMES)
    gps_csv = tmp_path / "gps_track.csv"
    write_gps_csv(gps_csv, N_FRAMES)
    cfg = make_cfg(tmp_path, video, gps_csv)

    scs = VideoDataset(cfg).scenarios()
    assert len(scs) == 1
    sc = scs[0]

    # name from filename stem; prior + intrinsics propagated from cfg.
    assert sc.name == video.stem
    assert sc.prior == (43.521955, -5.624290)
    assert sc.intrinsics.focal_px == 3713.0
    assert sc.meta["dataset"] == "video"
    assert sc.meta["source"] == video.stem

    samples = list(sc.samples())
    assert len(samples) == N_FRAMES
    for s in samples:
        assert isinstance(s.image_bgr, np.ndarray)
        assert s.gt is not None                 # populated from the CSV
        assert s.intrinsics.focal_px == 3713.0
        assert s.meta["source"] == video.stem

    # gt lat/lon come from the CSV via nearest-frame; frame 0 -> first row.
    first = samples[0]
    assert first.gt.frame == 0
    assert first.gt.lat == pytest.approx(43.5)
    assert first.gt.lon == pytest.approx(-5.6)

    # reference() is the shared FetchTile accessor — callable, but NOT called (no network).
    assert callable(sc.reference())


def test_samples_is_reiterable(tmp_path):
    video = make_video(tmp_path / "vid", N_FRAMES)
    gps_csv = tmp_path / "gps_track.csv"
    write_gps_csv(gps_csv, N_FRAMES)
    cfg = make_cfg(tmp_path, video, gps_csv)

    sc = VideoDataset(cfg).scenarios()[0]
    first_pass = list(sc.samples())
    second_pass = list(sc.samples())
    assert len(first_pass) == N_FRAMES
    assert len(second_pass) == N_FRAMES
    assert [s.frame_id for s in first_pass] == [s.frame_id for s in second_pass]


def test_missing_gps_track_yields_none_gt(tmp_path):
    video = make_video(tmp_path / "vid", N_FRAMES)
    missing_csv = tmp_path / "does_not_exist.csv"   # never written
    cfg = make_cfg(tmp_path, video, missing_csv)

    sc = VideoDataset(cfg).scenarios()[0]
    samples = list(sc.samples())
    assert len(samples) == N_FRAMES
    assert all(s.gt is None for s in samples)       # missing track must not crash
