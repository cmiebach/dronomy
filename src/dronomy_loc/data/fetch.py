"""Resumable, integrity-checked file download.

This is the data-acquisition stage that runs FIRST on a fresh machine: it pulls
the drone video (and, on Windows, a portable exiftool) onto the box so the rest
of the pipeline has something to ingest. A 3.7 GB download over a flaky link must
survive interruption, so the transfer is:

  * streamed to a `<dest>.part` sidecar (never a half-written final file),
  * **resumable** via an HTTP Range request (re-run continues, not restarts),
  * **size-verified** against `expected_bytes` before the atomic rename,
  * **idempotent** — a completed file (or a completed `.part`) is a no-op.

The HTTP layer is injected as `opener` so the byte-handling logic (resume offset,
size check, atomic commit) is unit-testable offline with a fake source.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable

# opener(url, start_byte) -> (status_code, headers, chunk_iterator)
# A 206 means the server honoured the Range (resume); a 200 means it ignored it
# and is sending the whole file from byte 0 (so any partial must be discarded).
Opener = Callable[[str, int], "tuple[int, dict, Iterable[bytes]]"]

_CHUNK = 1 << 20  # 1 MiB


def _requests_opener(timeout: float = 60.0) -> Opener:
    import requests

    def opener(url: str, start_byte: int):
        headers = {"Range": f"bytes={start_byte}-"} if start_byte > 0 else {}
        resp = requests.get(url, headers=headers, stream=True, timeout=timeout,
                            allow_redirects=True)
        resp.raise_for_status()
        return resp.status_code, dict(resp.headers), resp.iter_content(chunk_size=_CHUNK)

    return opener


def _content_length(headers: dict) -> int | None:
    for k in ("Content-Length", "content-length"):
        if k in headers:
            try:
                return int(headers[k])
            except (TypeError, ValueError):
                return None
    return None


def download_file(
    url: str,
    dest: str | Path,
    *,
    expected_bytes: int | None = None,
    opener: Opener | None = None,
    progress: Callable[[int, int | None], None] | None = None,
) -> Path:
    """Download `url` to `dest`, resuming a prior partial transfer if present.

    Returns `dest`. Raises `IOError` if `expected_bytes` is given and the final
    size doesn't match (a truncated/corrupt download is never promoted to the
    final path). `progress(downloaded, total_or_None)` is called as bytes land.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    # Already finished (final file the right size) -> nothing to do.
    if dest.exists() and (expected_bytes is None or dest.stat().st_size == expected_bytes):
        if expected_bytes is not None:
            if progress:
                progress(expected_bytes, expected_bytes)
            return dest
        # no expected size to check against; trust an existing final file
        if progress:
            progress(dest.stat().st_size, dest.stat().st_size)
        return dest

    # A complete `.part` (crash between last write and the rename) -> just commit.
    if expected_bytes is not None and part.exists() and part.stat().st_size == expected_bytes:
        os.replace(part, dest)
        if progress:
            progress(expected_bytes, expected_bytes)
        return dest

    opener = opener or _requests_opener()
    start = part.stat().st_size if part.exists() else 0
    if expected_bytes is not None and start > expected_bytes:
        start = 0  # corrupt/oversized partial -> restart clean

    status, headers, chunks = opener(url, start)
    mode = "ab"
    if start > 0 and status != 206:   # server ignored Range -> full body from 0
        start, mode = 0, "wb"

    total = expected_bytes
    if total is None:
        cl = _content_length(headers)
        total = (start + cl) if (cl is not None and status == 206) else cl

    written = start
    with open(part, mode) as fh:
        for c in chunks:
            if not c:
                continue
            fh.write(c)
            written += len(c)
            if progress:
                progress(written, total)

    if expected_bytes is not None and written != expected_bytes:
        raise IOError(
            f"size mismatch for {dest.name}: downloaded {written} bytes, "
            f"expected {expected_bytes} (left partial at {part.name} for resume)")
    os.replace(part, dest)
    return dest


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"
