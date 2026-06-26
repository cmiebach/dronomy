"""Recover the camera focal length in PIXELS — the one value that converts a
pixel-scale match into a metric ground footprint.

Most pipelines store focal length in millimetres, which is useless on its own:
the same lens projects different pixel counts on different sensors. DJI sidesteps
this by writing a pre-computed `CalibratedFocalLength` (already in pixels) into
the image XMP/warp header, so when it is present we take it verbatim (~3713 px
for the Mavic 3E wide camera). Otherwise we reconstruct it from whatever the EXIF
does give us, best source first:

  1) CalibratedFocalLength (DJI, px)            -> use directly (+ optical centre)
  2) FocalLength (mm) + sensor width (mm)       -> f_px = f_mm * W_px / sensor_mm
  3) FocalLengthIn35mmFormat (mm)               -> f_px = (f35 / 36) * W_px

(3) is the coarse fallback: a 35 mm-equivalent focal length is defined against a
36 mm-wide full frame, so dividing by 36 and scaling by the image width recovers
pixels without ever knowing the real sensor size.

This module is the SAMPLE-IMAGE path (needs exiftool + an actual frame). The
offline default the video adapter / runner use is `intrinsics_from_config`, which
just trusts the focal_px already in config.yaml — no exiftool, no image needed.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from ..framework.schema import CameraIntrinsics
from .telemetry import ExifToolNotFoundError, _run_exiftool, find_exiftool

# 35 mm-equivalent focal lengths are defined against a full-frame 36 mm width.
_FULL_FRAME_WIDTH_MM = 36.0

# FocalPlaneResolutionUnit -> millimetres per unit (EXIF tag 0xA210). With -n
# exiftool emits the numeric code; 2=inch, 3=cm, 4=mm are the values seen in DJI
# and consumer-camera headers. Resolution is "pixels per unit", so sensor width
# in mm = ImageWidth / FocalPlaneXResolution * (mm per unit).
_FOCAL_PLANE_UNIT_MM = {2: 25.4, 3: 10.0, 4: 1.0}


def _first(meta: dict, *keys):
    """First present, non-None value among `keys` (EXIF spells the same fact
    several ways: ImageWidth vs ExifImageWidth, etc.)."""
    for k in keys:
        v = meta.get(k)
        if v is not None:
            return v
    return None


def _image_width_px(meta: dict) -> float | None:
    w = _first(meta, "ImageWidth", "ExifImageWidth", "PixelXDimension")
    try:
        return float(w) if w is not None else None
    except (TypeError, ValueError):
        return None


def _sensor_width_mm(meta: dict, image_width_px: float | None) -> float | None:
    """Sensor width in mm from an explicit SensorWidth tag, else derived from the
    focal-plane resolution (pixels-per-unit) and the image width in pixels."""
    sw = _first(meta, "SensorWidth")
    if sw is not None:
        try:
            return float(sw)
        except (TypeError, ValueError):
            pass
    res = _first(meta, "FocalPlaneXResolution")
    unit = _first(meta, "FocalPlaneResolutionUnit")
    if res is None or image_width_px is None:
        return None
    try:
        res = float(res)
    except (TypeError, ValueError):
        return None
    if res <= 0:
        return None
    mm_per_unit = _FOCAL_PLANE_UNIT_MM.get(int(unit), 25.4) if unit is not None else 25.4
    # res is pixels-per-unit; width_in_units = W_px / res; width_mm = units * mm/unit.
    return image_width_px / res * mm_per_unit


def _principal_point(meta: dict) -> tuple[float, float] | None:
    cx = _first(meta, "CalibratedOpticalCenterX")
    cy = _first(meta, "CalibratedOpticalCenterY")
    if cx is None or cy is None:
        return None
    try:
        return (float(cx), float(cy))
    except (TypeError, ValueError):
        return None


def _hfov_deg(meta: dict) -> float | None:
    fov = _first(meta, "FOV")
    if fov is None:
        return None
    if isinstance(fov, (int, float)):
        return float(fov)
    # Some builds emit FOV as e.g. '83.9 deg' or '84.0 deg (0.50 m)'.
    tok = str(fov).strip().split()
    try:
        return float(tok[0]) if tok else None
    except ValueError:
        return None


def _focal_px(meta: dict) -> tuple[float, tuple[float, float] | None]:
    """Resolve focal length in pixels by the documented precedence. Returns
    (focal_px, principal_point); raises ValueError when no source is parseable."""
    # 1) DJI calibrated focal length: already in pixels, use as-is.
    cfl = _first(meta, "CalibratedFocalLength")
    if cfl is not None:
        try:
            return float(cfl), _principal_point(meta)
        except (TypeError, ValueError):
            pass

    width_px = _image_width_px(meta)

    # 2) Physical focal length (mm) scaled by the sensor sampling.
    f_mm = _first(meta, "FocalLength")
    if f_mm is not None and width_px is not None:
        sensor_mm = _sensor_width_mm(meta, width_px)
        if sensor_mm and sensor_mm > 0:
            try:
                f_mm = float(f_mm)
            except (TypeError, ValueError):
                f_mm = None
            if f_mm is not None:
                return f_mm * width_px / sensor_mm, _principal_point(meta)

    # 3) 35 mm-equivalent focal length against the 36 mm full-frame width.
    f35 = _first(meta, "FocalLengthIn35mmFormat", "FocalLengthIn35mmFilm")
    if f35 is not None and width_px is not None:
        try:
            return float(f35) / _FULL_FRAME_WIDTH_MM * width_px, _principal_point(meta)
        except (TypeError, ValueError):
            pass

    raise ValueError(
        "could not determine focal length in pixels: none of CalibratedFocalLength, "
        "FocalLength(mm)+sensor width, or FocalLengthIn35mmFormat+ImageWidth were "
        "present/parseable in the EXIF.")


def extract_intrinsics(
    path: str | Path,
    exiftool: str | None = None,
    timeout_s: int = 120,
) -> CameraIntrinsics:
    """Read camera intrinsics from a single image (or video) via exiftool.

    Runs `exiftool -j -n` and resolves focal_px by precedence (DJI calibrated px,
    then mm+sensor, then 35 mm-equivalent). `-n` keeps every value numeric so the
    arithmetic needs no unit-string parsing. Raises ExifToolNotFoundError when the
    binary is absent, RuntimeError on a nonzero exit, and ValueError when no focal
    source can be parsed."""
    exe = exiftool or find_exiftool()
    if not exe:
        raise ExifToolNotFoundError()

    cmd = [exe, "-j", "-n", str(path)]
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
    if not objs:
        raise ValueError(f"exiftool returned no metadata for {path}")
    meta = objs[0]

    focal_px, principal_point = _focal_px(meta)
    return CameraIntrinsics(
        focal_px=focal_px,
        principal_point=principal_point,
        hfov_deg=_hfov_deg(meta),
    )


def intrinsics_from_focal_px(
    focal_px: float,
    hfov_deg: float | None = None,
    principal_point: tuple[float, float] | None = None,
) -> CameraIntrinsics:
    """Trivial constructor for when the focal length in pixels is already known."""
    return CameraIntrinsics(
        focal_px=float(focal_px),
        principal_point=principal_point,
        hfov_deg=None if hfov_deg is None else float(hfov_deg),
    )


def intrinsics_from_config(cfg: SimpleNamespace) -> CameraIntrinsics:
    """Build intrinsics from the `camera` block of the loaded config (cfg.camera.
    focal_px / hfov_deg). This is the OFFLINE DEFAULT path: no exiftool, no image
    — the video adapter and runner trust the focal_px already in config.yaml."""
    cam = getattr(cfg, "camera", None)
    if cam is None:
        raise ValueError("config has no 'camera' section (need camera.focal_px)")
    focal_px = getattr(cam, "focal_px", None)
    if focal_px is None:
        raise ValueError("config camera section has no focal_px")
    return intrinsics_from_focal_px(
        focal_px=focal_px,
        hfov_deg=getattr(cam, "hfov_deg", None),
    )
