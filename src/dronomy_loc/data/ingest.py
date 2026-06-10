"""Sharded, resumable, integrity-verified video ingestion.

`extract_frames` is fine for experiments, but a full flight is a long decode
and one bad write (disk full, crash, Ctrl-C) silently poisons the frame set.
Here ingestion is restartable and self-checking: frames are grouped into fixed
time shards, every JPEG is hashed and re-read at write time, and a manifest is
committed atomically each time a shard completes. A re-run replays the SAME
deterministic sampling pass (sequential grab/step — no POS_* seeking, which is
flaky on Windows) and skips re-encoding shards already marked complete, so an
interrupted or damaged ingest repairs itself; the resume savings are the JPEG
encodes/writes, not decode time, which is acceptable and safe.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from . import frames


class IngestMismatchError(RuntimeError):
    """The existing ingest directory belongs to a different video or settings."""


@dataclass(frozen=True)
class ShardSpec:
    """Fixed time shards: shard id = floor(t_seconds / shard_seconds)."""
    shard_seconds: float = 30.0

    def shard_id(self, t_seconds: float) -> int:
        return int(t_seconds // self.shard_seconds)

    def dir_for(self, out_dir: str | Path, shard_id: int) -> Path:
        return Path(out_dir) / "shards" / f"shard_{shard_id:04d}"


@dataclass
class IngestResult:
    manifest_path: Path
    n_shards: int
    n_frames_written: int
    n_frames_skipped: int
    completed: bool


@dataclass
class VerifyReport:
    ok: bool
    n_checked: int
    problems: list[str] = field(default_factory=list)


def _settings_hash(settings: dict) -> str:
    return hashlib.sha1(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def _load_manifest(out_dir: Path) -> dict | None:
    path = Path(out_dir) / "manifest.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_manifest(out_dir: Path, manifest: dict) -> Path:
    # tmp + os.replace: a crash mid-write can never leave a torn manifest.
    path = Path(out_dir) / "manifest.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    os.replace(tmp, path)
    return path


def _write_verified_jpeg(path: Path, image, quality: int) -> tuple[int, str]:
    """imencode + tofile (cv2.imwrite fails silently on non-ASCII Windows
    paths), then re-read and decode to prove the bytes on disk are valid."""
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise IOError(f"JPEG encode failed: {path}")
    data = buf.tobytes()
    buf.tofile(str(path))
    back = np.fromfile(str(path), dtype=np.uint8)
    sha = hashlib.sha1(data).hexdigest()
    if (back.nbytes != len(data)
            or hashlib.sha1(back.tobytes()).hexdigest() != sha
            or cv2.imdecode(back, cv2.IMREAD_COLOR) is None):
        raise IOError(f"Frame failed write verification: {path}")
    return len(data), sha


def _write_frames_csv(out_dir: Path, manifest: dict) -> Path:
    """Flatten the manifest for downstream steps (the GPS ground-truth track
    joins on "index"). Written only after a fully completed run; tmp+os.replace
    so a crash mid-write can never leave a torn CSV."""
    path = Path(out_dir) / "frames.csv"
    tmp = path.with_suffix(".csv.tmp")
    cols = ["shard", "index", "t_ms", "blur_score", "filename", "bytes", "sha1"]
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for sid in sorted(manifest["shards"], key=int):
            for fr in manifest["shards"][sid]["frames"]:
                w.writerow({"shard": int(sid), **fr})
    os.replace(tmp, path)
    return path


def ingest_video(
    video_path: str | Path,
    out_dir: str | Path,
    *,
    every_n_seconds: float = 1.0,
    shard_seconds: float = 30.0,
    blur_filter: str = "sharpest",
    min_blur_var: float = 0.0,
    resize_long_edge: int | None = 1920,
    jpeg_quality: int = 95,
    max_frames: int | None = None,
    force: bool = False,
) -> IngestResult:
    """Ingest the video into `out_dir/shards/shard_NNNN/frame_NNNNNN_tNNNNNNNNms.jpg`.

    Single sequential pass over the source iterators from `frames` (no video
    seeking). If a matching manifest already exists, shards marked "complete"
    are skipped (frames still iterate, just not re-encoded), so re-running
    after a crash, a `max_frames` cap, or a `verify_ingest` demotion repairs
    exactly the missing/partial shards. A manifest for a different video or
    different settings raises `IngestMismatchError` unless `force=True`, which
    wipes the ingest and restarts. `max_frames` caps the frames WRITTEN this
    run (skipped frames of complete shards don't count, so re-running with the
    same cap always makes progress); hitting it leaves the manifest resumable
    (completed stays false, the in-flight shard stays "partial")."""
    video_path, out_dir = Path(video_path), Path(out_dir)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = {"every_n_seconds": every_n_seconds, "shard_seconds": shard_seconds,
                "blur_filter": blur_filter, "min_blur_var": min_blur_var,
                "resize_long_edge": resize_long_edge, "jpeg_quality": jpeg_quality}
    s_hash = _settings_hash(settings)
    meta = frames.probe(video_path)
    video_info = {"name": video_path.name, "size_bytes": video_path.stat().st_size,
                  "n_frames": meta["n_frames"], "fps": meta["fps"] or 30.0,
                  "width": meta["width"], "height": meta["height"]}

    manifest = _load_manifest(out_dir)
    if manifest is not None:
        old_video = manifest.get("video", {})
        mismatches = [f"video.{k}: manifest={old_video.get(k)!r} actual={video_info[k]!r}"
                      for k in ("name", "size_bytes", "n_frames")
                      if old_video.get(k) != video_info[k]]
        if manifest.get("settings_hash") != s_hash:
            mismatches.append("settings changed")
        if mismatches:
            if not force:
                raise IngestMismatchError(
                    f"{out_dir} holds a different ingest ({'; '.join(mismatches)}). "
                    "Pass force=True to wipe it and restart.")
            shutil.rmtree(out_dir / "shards", ignore_errors=True)
            (out_dir / "frames.csv").unlink(missing_ok=True)
            (out_dir / "manifest.json").unlink(missing_ok=True)
            manifest = None

    if manifest is None:
        manifest = {"video": video_info, "settings": settings, "settings_hash": s_hash,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "shards": {}, "completed": False}

    shards = manifest["shards"]
    complete_ids = {sid for sid, sh in shards.items() if sh["status"] == "complete"}
    if manifest["completed"] and complete_ids == set(shards):
        if not (out_dir / "frames.csv").exists():  # lost between commit and write
            _write_frames_csv(out_dir, manifest)
        n_done = sum(len(sh["frames"]) for sh in shards.values())
        return IngestResult(out_dir / "manifest.json", len(shards), 0, n_done, True)

    if blur_filter == "sharpest":
        source = frames.iter_frames_sharpest(video_path, every_n_seconds=every_n_seconds,
                                             resize_long_edge=resize_long_edge,
                                             min_blur_var=min_blur_var)
    elif blur_filter == "off":
        source = frames.iter_frames(video_path, every_n_seconds=every_n_seconds,
                                    resize_long_edge=resize_long_edge)
    else:
        raise ValueError(f"Unknown blur_filter: {blur_filter!r} (use 'sharpest' or 'off')")

    spec = ShardSpec(shard_seconds)
    written = skipped = 0
    capped = False
    current: str | None = None
    rebuilt: set[str] = set()  # shards (re)written by THIS run

    for fi in source:
        sid = str(spec.shard_id(fi.t_seconds))
        if sid != current:
            if current in rebuilt:  # shard boundary: commit the finished shard
                shards[current]["status"] = "complete"
                _save_manifest(out_dir, manifest)
            current = sid
        if sid in complete_ids:
            skipped += 1
        else:
            if sid not in rebuilt:  # first frame of a fresh/partial shard
                shards[sid] = {"status": "partial", "frames": []}
                rebuilt.add(sid)
                spec.dir_for(out_dir, int(sid)).mkdir(parents=True, exist_ok=True)
            t_ms = int(fi.t_seconds * 1000)
            name = f"frame_{fi.index:06d}_t{t_ms:08d}ms.jpg"
            n_bytes, sha = _write_verified_jpeg(spec.dir_for(out_dir, int(sid)) / name,
                                                fi.image, jpeg_quality)
            shards[sid]["frames"].append({"index": fi.index, "t_ms": t_ms,
                                          "blur_score": round(fi.blur_score, 2),
                                          "filename": name, "bytes": n_bytes, "sha1": sha})
            written += 1
        if max_frames and written >= max_frames:
            capped = True  # cap mid-shard: leave it "partial", resumable
            break

    if not capped:
        if current in rebuilt:  # the video ran to the end, so the tail shard is whole
            shards[current]["status"] = "complete"
        manifest["completed"] = True
    if manifest["completed"]:
        # frames.csv BEFORE the completed=true commit: a crash between the two
        # must leave an incomplete manifest (self-repairs), never a missing CSV.
        _write_frames_csv(out_dir, manifest)
    manifest_path = _save_manifest(out_dir, manifest)
    return IngestResult(manifest_path, len(shards), written, skipped, manifest["completed"])


def verify_ingest(out_dir: str | Path) -> VerifyReport:
    """Re-read every manifest-listed frame: existence, size, sha1, decodability.

    Damaged shards are demoted to "partial" (and `completed` cleared) so the
    next `ingest_video` run re-writes exactly those shards — that demotion IS
    the repair hook."""
    out_dir = Path(out_dir)
    manifest = _load_manifest(out_dir)
    if manifest is None:
        return VerifyReport(ok=False, n_checked=0,
                            problems=[f"no manifest.json in {out_dir}"])
    problems: list[str] = []
    n_checked = 0
    demoted = False
    for sid, shard in manifest["shards"].items():
        sdir = ShardSpec().dir_for(out_dir, int(sid))
        shard_ok = True
        for fr in shard["frames"]:
            n_checked += 1
            rel = f"shards/shard_{int(sid):04d}/{fr['filename']}"
            path = sdir / fr["filename"]
            if not path.exists():
                problems.append(f"missing: {rel}")
                shard_ok = False
                continue
            data = np.fromfile(str(path), dtype=np.uint8)
            if data.nbytes != fr["bytes"]:
                problems.append(f"size mismatch ({data.nbytes} != {fr['bytes']}): {rel}")
                shard_ok = False
                continue
            if hashlib.sha1(data.tobytes()).hexdigest() != fr["sha1"]:
                problems.append(f"sha1 mismatch: {rel}")
                shard_ok = False
                continue
            if cv2.imdecode(data, cv2.IMREAD_COLOR) is None:
                problems.append(f"undecodable: {rel}")
                shard_ok = False
        if not shard_ok and shard["status"] == "complete":
            shard["status"] = "partial"
            demoted = True
    if manifest["completed"] and not (out_dir / "frames.csv").exists():
        problems.append("missing: frames.csv (re-run ingest to regenerate)")
    if demoted:
        manifest["completed"] = False
        _save_manifest(out_dir, manifest)
    return VerifyReport(ok=not problems, n_checked=n_checked, problems=problems)
