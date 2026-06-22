"""Extract the per-frame GPS track DJI embeds in the video's `djmd` metadata stream.

GROUND TRUTH ONLY: this track is never an input to the localizer — the project is
telemetry-free by design. It exists solely to score the estimated trajectory
against where the drone actually was.

exiftool decodes the embedded samples (`exiftool -ee -j -n -G3 ...`). The Mavic 3
Enterprise logs one sample per video frame, so the DocN group number maps directly
to a frame index: frame = N - 1. Invalid fixes ((0,0) or out-of-range) are dropped
but surviving fixes KEEP their doc-derived frame numbers — no reindexing — so they
stay aligned with the video.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GPSFix:
    frame: int             # video frame index (one telemetry sample per frame)
    t_s: float | None      # SampleTime in seconds, when the stream provides it
    lat: float
    lon: float
    alt_m: float | None


class ExifToolNotFoundError(RuntimeError):
    def __init__(self):
        super().__init__(
            "exiftool not found. Install it with:\n"
            "    winget install -e --id OliverBetz.ExifTool\n"
            "or download from https://exiftool.org (rename to exiftool.exe, put it "
            "on PATH or set the EXIFTOOL environment variable to its full path)."
        )


def find_exiftool() -> str | None:
    """Locate exiftool: EXIFTOOL env var, then PATH, then common Windows spots."""
    env = os.environ.get("EXIFTOOL")
    if env and Path(env).is_file():
        return env
    found = shutil.which("exiftool")
    if found:
        return found
    candidates = [
        Path("C:/Windows/exiftool.exe"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WindowsApps/exiftool.exe",
        Path("C:/Program Files/exiftool/exiftool.exe"),
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


# ── exiftool JSON parsing ─────────────────────────────────────────────

_DOC_KEY = re.compile(r"^Doc(\d+):(.*)$")  # -G3 prefixes embedded samples as Doc<N>:


def _parse_sample_time(v) -> float | None:
    # SampleTime arrives numeric with -n, but some builds emit '1.234 s'.
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s.endswith("s"):
        s = s[:-1].strip()
    try:
        return float(s)
    except ValueError:
        return None


def _valid(lat: float, lon: float) -> bool:
    return (lat, lon) != (0.0, 0.0) and -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def _parse_exiftool_json(objs: list[dict]) -> tuple[list[GPSFix], int]:
    """Returns (clean fixes, n dropped). Handles both shapes -ee can produce:
    Doc<N>:-prefixed keys in one object (-G3), or one sample per array element.
    In the flat shape every sample-bearing element consumes a frame slot even
    when its fix is missing/invalid — compacting would silently shift every
    later frame index and corrupt the ground-truth join."""
    groups: dict[int, dict] = {}
    flat: list[tuple[int, dict]] = []  # (element position, sample)
    flat_pos = 0
    for obj in objs:
        sample: dict = {}
        for key, val in obj.items():
            m = _DOC_KEY.match(key)
            target = groups.setdefault(int(m.group(1)), {}) if m else sample
            tag = m.group(2) if m else key
            if tag.endswith("GPSLatitude"):
                target["lat"] = val
            elif tag.endswith("GPSLongitude"):
                target["lon"] = val
            elif tag.endswith("GPSAltitude"):
                target["alt"] = val
            elif tag.endswith("SampleTime"):
                target["t"] = val
        if sample:  # any telemetry tag at the top level = a sample element
            flat.append((flat_pos, sample))
            flat_pos += 1

    if groups:
        ordered = [(n - 1, groups[n]) for n in sorted(groups)]  # frame = DocN - 1
    else:
        ordered = flat

    fixes: list[GPSFix] = []
    dropped = 0
    for frame, s in ordered:
        try:
            lat, lon = float(s["lat"]), float(s["lon"])
        except (KeyError, TypeError, ValueError):
            dropped += 1
            continue
        if not _valid(lat, lon):
            dropped += 1
            continue
        alt = s.get("alt")
        fixes.append(GPSFix(frame=frame, t_s=_parse_sample_time(s.get("t")),
                            lat=lat, lon=lon,
                            alt_m=float(alt) if alt is not None else None))
    return fixes, dropped


# ── extraction ────────────────────────────────────────────────────────

def _run_exiftool(cmd: list[str], timeout_s: int) -> subprocess.CompletedProcess:
    """`subprocess.run(timeout=...)` is not enough on Windows: exiftool.exe is a
    launcher whose perl child inherits the stdout/stderr pipes, so after the
    timeout kills only the launcher the orphan keeps the pipes open and
    communicate() blocks forever. Kill the whole process tree instead."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace")
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True)
        else:
            proc.kill()
        proc.communicate()
        raise RuntimeError(f"exiftool timed out after {timeout_s}s") from None
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def extract_gps_track(
    video_path: str | Path,
    out_csv: str | Path | None = None,
    exiftool: str | None = None,
    timeout_s: int = 600,
) -> list[GPSFix]:
    """Run exiftool on the video and return the clean per-frame GPS track."""
    exe = exiftool or find_exiftool()
    if not exe:
        raise ExifToolNotFoundError()

    cmd = [exe, "-ee", "-j", "-n", "-G3", "-api", "largefilesupport=1", str(video_path)]
    proc = _run_exiftool(cmd, timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(f"exiftool failed (rc={proc.returncode}): "
                           f"{(proc.stderr or '').strip()[:500]}")
    try:
        objs = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"exiftool produced unparseable JSON: {e}") from e
    if isinstance(objs, dict):
        objs = [objs]

    fixes, dropped = _parse_exiftool_json(objs)
    if dropped:
        print(f"telemetry: dropped {dropped} invalid GPS fixes")
    if out_csv:
        write_track_csv(fixes, out_csv)
    return fixes


# ── CSV persistence + ground-truth lookup ─────────────────────────────

def write_track_csv(fixes: list[GPSFix], out_csv: str | Path) -> Path:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "t_s", "lat", "lon", "alt_m"])
        for f in fixes:
            w.writerow([f.frame,
                        "" if f.t_s is None else f.t_s,
                        f.lat, f.lon,
                        "" if f.alt_m is None else f.alt_m])
    return out_csv


def load_track_csv(path: str | Path) -> list[GPSFix]:
    fixes: list[GPSFix] = []
    with open(path, "r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            fixes.append(GPSFix(
                frame=int(row["frame"]),
                t_s=float(row["t_s"]) if row["t_s"] else None,
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                alt_m=float(row["alt_m"]) if row["alt_m"] else None,
            ))
    return fixes


def gt_for_frame(fixes: list[GPSFix], frame_idx: int) -> GPSFix:
    """Ground-truth fix for a frame: exact when present, else nearest by frame."""
    if not fixes:
        raise ValueError("empty GPS track")
    return min(fixes, key=lambda f: abs(f.frame - frame_idx))
