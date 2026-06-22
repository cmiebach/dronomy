"""Read the drone video and extract frames with OpenCV.

Per the brief: treat each frame independently for now. This module both streams
frames lazily (`iter_frames`) and writes a sampled set to disk (`extract_frames`).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2


@dataclass
class FrameInfo:
    index: int                 # frame index in the source video
    t_seconds: float           # timestamp in seconds
    image: "object"            # numpy.ndarray (BGR), or None when only metadata is needed
    blur_score: float = 0.0    # variance-of-Laplacian focus measure (higher = sharper)


def _resize_long_edge(img, long_edge: int | None):
    if not long_edge:
        return img
    h, w = img.shape[:2]
    if max(h, w) <= long_edge:
        return img
    scale = long_edge / max(h, w)
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def focus_measure(img, work_edge: int = 960) -> float:
    """Variance of the Laplacian — a standard sharpness/focus score.

    Higher means sharper; motion-blurred frames score low. The frame is
    downscaled to ``work_edge`` first so the measure is fast and comparable
    across frames of the same source resolution (we only ever rank frames of
    one video against each other, so the absolute scale doesn't matter)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = _resize_long_edge(gray, work_edge)
    return float(cv2.Laplacian(small, cv2.CV_64F).var())


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


def iter_frames_sharpest(
    video_path: str | Path,
    every_n_seconds: float = 1.0,
    max_frames: int | None = None,
    resize_long_edge: int | None = None,
    min_blur_var: float = 0.0,
) -> Iterator[FrameInfo]:
    """Yield the SHARPEST frame from each sampling window (~`every_n_seconds`).

    Unlike `iter_frames`, this decodes every frame (sharpness needs the pixels),
    scores each by `focus_measure`, and emits only the best one per window. This
    drops motion-blurred frames without a fragile absolute threshold while still
    yielding ~1 frame per window, so spatial coverage stays uniform. A window is
    skipped entirely only if even its sharpest frame is below `min_blur_var`
    (default 0 = never skip)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps * every_n_seconds))
    idx, emitted = 0, 0
    best: FrameInfo | None = None

    try:
        while True:
            ok, frame = cap.read()         # must decode: sharpness needs pixels
            if not ok:
                break
            score = focus_measure(frame)
            if best is None or score > best.blur_score:
                best = FrameInfo(index=idx, t_seconds=idx / fps, image=frame,
                                 blur_score=score)
            if (idx + 1) % step == 0 and best is not None:
                if best.blur_score >= min_blur_var:
                    best.image = _resize_long_edge(best.image, resize_long_edge)
                    yield best
                    emitted += 1
                    if max_frames and emitted >= max_frames:
                        return
                best = None
            idx += 1
        # tail window (partial), if anything is pending
        if best is not None and best.blur_score >= min_blur_var and (
                not max_frames or emitted < max_frames):
            best.image = _resize_long_edge(best.image, resize_long_edge)
            yield best
    finally:
        cap.release()


def extract_frames(
    video_path: str | Path,
    out_dir: str | Path,
    every_n_seconds: float = 1.0,
    max_frames: int | None = None,
    resize_long_edge: int | None = None,
    jpeg_quality: int = 95,
    blur_filter: str = "sharpest",
    min_blur_var: float = 0.0,
    write_manifest: bool = True,
) -> list[Path]:
    """Extract sampled frames to JPGs named `frame_<index:06d>_t<ms>.jpg`.

    `blur_filter`:
      * "sharpest" (default) — keep the sharpest frame per window (drops blur).
      * "off"                — plain uniform sampling (fast grab/skip, no decode-all).

    Also writes a `frames.csv` manifest (index, t_ms, blur_score, filename) so
    downstream steps can join frames to the GPS ground-truth track by index and
    inspect the focus scores. Returns the list of written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if blur_filter == "sharpest":
        source = iter_frames_sharpest(video_path, every_n_seconds, max_frames,
                                      resize_long_edge, min_blur_var)
    elif blur_filter == "off":
        source = iter_frames(video_path, every_n_seconds, max_frames, resize_long_edge)
    else:
        raise ValueError(f"Unknown blur_filter: {blur_filter!r} (use 'sharpest' or 'off')")

    written: list[Path] = []
    rows: list[dict] = []
    for fi in source:
        name = f"frame_{fi.index:06d}_t{int(fi.t_seconds * 1000):07d}ms.jpg"
        path = out_dir / name
        cv2.imwrite(str(path), fi.image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        written.append(path)
        rows.append({"index": fi.index, "t_ms": int(fi.t_seconds * 1000),
                     "blur_score": round(fi.blur_score, 2), "filename": name})

    if write_manifest and rows:
        with open(out_dir / "frames.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["index", "t_ms", "blur_score", "filename"])
            w.writeheader()
            w.writerows(rows)
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
