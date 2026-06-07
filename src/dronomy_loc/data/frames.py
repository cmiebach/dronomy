"""Read the drone video and extract frames with OpenCV.

Per the brief: treat each frame independently for now. This module both streams
frames lazily (`iter_frames`) and writes a sampled set to disk (`extract_frames`).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2


@dataclass
class FrameInfo:
    index: int          # frame index in the source video
    t_seconds: float    # timestamp in seconds
    image: "object"     # numpy.ndarray (BGR), or None when only metadata is needed


def _resize_long_edge(img, long_edge: int | None):
    if not long_edge:
        return img
    h, w = img.shape[:2]
    if max(h, w) <= long_edge:
        return img
    scale = long_edge / max(h, w)
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def iter_frames(
    video_path: str | Path,
    every_n_seconds: float = 1.0,
    max_frames: int | None = None,
    resize_long_edge: int | None = None,
) -> Iterator[FrameInfo]:
    """Yield sampled frames lazily. Uses a frame-step derived from the video FPS
    (robust on Windows where seeking by POS_MSEC can be flaky)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps * every_n_seconds))
    idx, emitted = 0, 0
    try:
        while True:
            ok = cap.grab()            # cheap: advance without decoding
            if not ok:
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                frame = _resize_long_edge(frame, resize_long_edge)
                yield FrameInfo(index=idx, t_seconds=idx / fps, image=frame)
                emitted += 1
                if max_frames and emitted >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()


def extract_frames(
    video_path: str | Path,
    out_dir: str | Path,
    every_n_seconds: float = 1.0,
    max_frames: int | None = None,
    resize_long_edge: int | None = None,
    jpeg_quality: int = 95,
) -> list[Path]:
    """Extract sampled frames to JPGs named `frame_<index:06d>_t<ms>.jpg`.
    Returns the list of written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fi in iter_frames(video_path, every_n_seconds, max_frames, resize_long_edge):
        name = f"frame_{fi.index:06d}_t{int(fi.t_seconds * 1000):07d}ms.jpg"
        path = out_dir / name
        cv2.imwrite(str(path), fi.image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        written.append(path)
    return written


def probe(video_path: str | Path) -> dict:
    """Return basic video properties (no ffprobe needed)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "n_frames": n, "width": w, "height": h,
            "duration_s": (n / fps) if fps else None}
