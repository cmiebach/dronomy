"""Ingest pipeline tests — fully offline, synthetic videos only (cv2.VideoWriter)."""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronomy_loc.data import ingest  # noqa: E402

# fps=10, every=0.1s -> step 1 (all frames); 0.5s shards -> 5 frames per shard.
ARGS = dict(every_n_seconds=0.1, shard_seconds=0.5, blur_filter="off",
            resize_long_edge=None, jpeg_quality=90)


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


def load_manifest(out: Path) -> dict:
    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))


def test_basic_ingest(tmp_path):
    video = make_video(tmp_path / "vid", 20)
    out = tmp_path / "ingest"
    res = ingest.ingest_video(video, out, **ARGS)
    assert res.completed
    assert res.n_frames_written == 20
    assert res.n_frames_skipped == 0
    assert res.n_shards == 4

    m = load_manifest(out)
    assert m["completed"] is True
    assert set(m["shards"]) == {"0", "1", "2", "3"}
    assert all(s["status"] == "complete" for s in m["shards"].values())
    assert m["video"]["n_frames"] == 20

    with open(out / "frames.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 20
    for r in rows:
        f = out / "shards" / f"shard_{int(r['shard']):04d}" / r["filename"]
        assert f.exists() and f.stat().st_size == int(r["bytes"])

    # No-op re-run: everything already complete, nothing rewritten.
    res2 = ingest.ingest_video(video, out, **ARGS)
    assert res2.completed and res2.n_frames_written == 0 and res2.n_frames_skipped == 20


def test_resume_repairs_only_damaged_shard(tmp_path):
    video = make_video(tmp_path / "vid", 20)
    out = tmp_path / "ingest"
    ingest.ingest_video(video, out, **ARGS)
    m = load_manifest(out)
    victim_fr = m["shards"]["1"]["frames"][2]
    victim = out / "shards" / "shard_0001" / victim_fr["filename"]
    others = {p: p.stat().st_mtime_ns for p in (out / "shards").rglob("*.jpg")
              if p.parent.name != "shard_0001"}

    victim.unlink()
    rep = ingest.verify_ingest(out)
    assert not rep.ok and any("missing" in p for p in rep.problems)
    m2 = load_manifest(out)
    assert m2["shards"]["1"]["status"] == "partial"
    assert m2["completed"] is False

    res = ingest.ingest_video(video, out, **ARGS)
    assert res.completed
    assert res.n_frames_written == 5      # only the demoted shard was rebuilt
    assert res.n_frames_skipped == 15
    assert victim.exists()
    m3 = load_manifest(out)
    assert m3["shards"]["1"]["frames"][2]["sha1"] == victim_fr["sha1"]  # deterministic
    for p, mtime in others.items():
        assert p.stat().st_mtime_ns == mtime  # untouched shards were not rewritten
    assert ingest.verify_ingest(out).ok


def test_settings_mismatch_raises_and_force_recovers(tmp_path):
    video = make_video(tmp_path / "vid", 10)
    out = tmp_path / "ingest"
    ingest.ingest_video(video, out, **ARGS)

    changed = {**ARGS, "jpeg_quality": 70}
    with pytest.raises(ingest.IngestMismatchError, match="settings changed"):
        ingest.ingest_video(video, out, **changed)

    res = ingest.ingest_video(video, out, force=True, **changed)
    assert res.completed and res.n_frames_written == 10 and res.n_frames_skipped == 0
    assert load_manifest(out)["settings"]["jpeg_quality"] == 70


def test_verify_catches_corrupt_jpg(tmp_path):
    video = make_video(tmp_path / "vid", 10)
    out = tmp_path / "ingest"
    ingest.ingest_video(video, out, **ARGS)
    m = load_manifest(out)
    fr = m["shards"]["0"]["frames"][0]
    f = out / "shards" / "shard_0000" / fr["filename"]
    f.write_bytes(f.read_bytes()[: fr["bytes"] // 2])  # truncate

    rep = ingest.verify_ingest(out)
    assert not rep.ok
    assert rep.n_checked == 10
    assert any(fr["filename"] in p for p in rep.problems)

    res = ingest.ingest_video(video, out, **ARGS)  # repair pass
    assert res.completed and res.n_frames_written == 5
    assert ingest.verify_ingest(out).ok


def test_max_frames_cap_then_resume(tmp_path):
    video = make_video(tmp_path / "vid", 20)
    out = tmp_path / "ingest"
    res1 = ingest.ingest_video(video, out, max_frames=7, **ARGS)
    assert not res1.completed
    assert res1.n_frames_written == 7
    m = load_manifest(out)
    assert m["completed"] is False
    assert m["shards"]["0"]["status"] == "complete"
    assert m["shards"]["1"]["status"] == "partial"  # cap hit mid-shard
    assert not (out / "frames.csv").exists()

    res2 = ingest.ingest_video(video, out, **ARGS)
    assert res2.completed
    assert res2.n_frames_skipped == 5     # shard 0 kept
    assert res2.n_frames_written == 15    # shard 1 rebuilt + shards 2-3
    with open(out / "frames.csv", newline="", encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 20


def test_constant_max_frames_resume_terminates(tmp_path):
    """Re-running with the SAME cap must make progress each run (the cap counts
    frames WRITTEN, not frames replayed from already-complete shards)."""
    video = make_video(tmp_path / "vid", 20)
    out = tmp_path / "ingest"
    runs = 0
    while True:
        res = ingest.ingest_video(video, out, max_frames=7, **ARGS)
        runs += 1
        assert runs <= 5, "constant-cap resume made no progress (livelock)"
        if res.completed:
            break
        assert res.n_frames_written > 0  # every capped run advances
    with open(out / "frames.csv", newline="", encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 20


def test_frames_csv_regenerated_if_lost_after_completion(tmp_path):
    """frames.csv is committed before completed=true; if it is lost anyway
    (crash window, manual delete), verify flags it and a re-run restores it."""
    video = make_video(tmp_path / "vid", 10)
    out = tmp_path / "ingest"
    ingest.ingest_video(video, out, **ARGS)
    (out / "frames.csv").unlink()

    rep = ingest.verify_ingest(out)
    assert not rep.ok and any("frames.csv" in p for p in rep.problems)

    res = ingest.ingest_video(video, out, **ARGS)  # early-return path regenerates
    assert res.completed and res.n_frames_written == 0
    with open(out / "frames.csv", newline="", encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 10
    assert ingest.verify_ingest(out).ok


def test_blur_filter_modes(tmp_path):
    video = make_video(tmp_path / "vid", 20)
    for mode in ("off", "sharpest"):
        out = tmp_path / f"ingest_{mode}"
        res = ingest.ingest_video(video, out, every_n_seconds=0.2, shard_seconds=0.5,
                                  blur_filter=mode, resize_long_edge=None, jpeg_quality=90)
        assert res.completed
        assert res.n_frames_written == 10  # 10 windows of 2 frames each
        m = load_manifest(out)
        scores = [fr["blur_score"] for sh in m["shards"].values() for fr in sh["frames"]]
        if mode == "sharpest":
            assert all(s > 0 for s in scores)


def test_missing_video_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest.ingest_video(tmp_path / "nope.mp4", tmp_path / "out")


def test_non_ascii_out_dir(tmp_path):
    video = make_video(tmp_path / "vid", 10)
    out = tmp_path / "área_42" / "ingest"
    res = ingest.ingest_video(video, out, **ARGS)
    assert res.completed and res.n_frames_written == 10
    assert ingest.verify_ingest(out).ok
    fr = load_manifest(out)["shards"]["0"]["frames"][0]
    data = np.fromfile(str(out / "shards" / "shard_0000" / fr["filename"]), dtype=np.uint8)
    assert cv2.imdecode(data, cv2.IMREAD_COLOR) is not None


def test_empty_video_clean_result(tmp_path):
    # Zero-frame mp4s don't even open; MJPG/.avi yields an openable 0-frame clip.
    p = tmp_path / "empty.avi"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    if not vw.isOpened():
        pytest.skip("MJPG writer unavailable")
    vw.release()
    cap = cv2.VideoCapture(str(p))
    opened = cap.isOpened()
    cap.release()
    if not opened:
        pytest.skip("backend cannot reopen a zero-frame clip")

    res = ingest.ingest_video(p, tmp_path / "out", **ARGS)
    assert res.completed
    assert res.n_shards == 0 and res.n_frames_written == 0


def test_video_shorter_than_one_window(tmp_path):
    video = make_video(tmp_path / "vid", 3)  # 0.3 s of video, 1 s windows
    res = ingest.ingest_video(video, tmp_path / "out", every_n_seconds=1.0,
                              shard_seconds=30.0, blur_filter="sharpest",
                              resize_long_edge=None, jpeg_quality=90)
    assert res.completed
    assert res.n_shards == 1 and res.n_frames_written == 1  # tail window


def test_fps_zero_falls_back_to_30(tmp_path, monkeypatch):
    video = make_video(tmp_path / "vid", 5)
    real_probe = ingest.frames.probe

    def fake_probe(p):
        m = real_probe(p)
        m["fps"] = 0.0
        return m

    monkeypatch.setattr(ingest.frames, "probe", fake_probe)
    res = ingest.ingest_video(video, tmp_path / "out", **ARGS)
    assert res.completed
    assert load_manifest(tmp_path / "out")["video"]["fps"] == 30.0
