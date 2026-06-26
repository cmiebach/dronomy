"""Camera intrinsics extraction — all offline, exiftool is mocked.

The focal-length-in-pixels precedence is what these tests pin down: DJI's
calibrated px wins, then mm+sensor arithmetic, then the 35 mm-equivalent
fallback. The Popen/CompletedProcess mock mirrors tests/test_telemetry.py."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronomy_loc.data import intrinsics  # noqa: E402
from dronomy_loc.data.intrinsics import (  # noqa: E402
    extract_intrinsics, intrinsics_from_config, intrinsics_from_focal_px,
)
from dronomy_loc.data.telemetry import ExifToolNotFoundError  # noqa: E402
from dronomy_loc.framework.schema import CameraIntrinsics  # noqa: E402


def _mock_exiftool(monkeypatch, payload=None, returncode=0, stderr=""):
    stdout = json.dumps(payload) if payload is not None else ""

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            self.pid = 4242
            self.returncode = None

        def communicate(self, timeout=None):
            self.returncode = returncode
            return stdout, stderr

    monkeypatch.setattr(intrinsics.subprocess, "Popen", FakePopen)


def test_calibrated_focal_length_used_directly(monkeypatch):
    # DJI Mavic 3E wide camera: CalibratedFocalLength already in px, optical
    # centre present -> principal_point parsed too.
    payload = [{
        "SourceFile": "DJI_0001.JPG",
        "CalibratedFocalLength": 3713.0,
        "CalibratedOpticalCenterX": 2736.0,
        "CalibratedOpticalCenterY": 1824.0,
        "FOV": 84.0,
        # decoys that MUST be ignored when calibrated px is present:
        "FocalLength": 12.29, "FocalLengthIn35mmFormat": 24.0, "ImageWidth": 5472,
    }]
    _mock_exiftool(monkeypatch, payload)
    ci = extract_intrinsics("DJI_0001.JPG", exiftool="exiftool")
    assert isinstance(ci, CameraIntrinsics)
    assert ci.focal_px == pytest.approx(3713.0)
    assert ci.principal_point == pytest.approx((2736.0, 1824.0))
    assert ci.hfov_deg == pytest.approx(84.0)


def test_calibrated_focal_without_optical_center(monkeypatch):
    payload = [{"SourceFile": "x.JPG", "CalibratedFocalLength": 3713.0}]
    _mock_exiftool(monkeypatch, payload)
    ci = extract_intrinsics("x.JPG", exiftool="exiftool")
    assert ci.focal_px == pytest.approx(3713.0)
    assert ci.principal_point is None
    assert ci.hfov_deg is None


def test_focal_mm_plus_sensor_width(monkeypatch):
    # No calibrated px -> f_px = f_mm * W_px / sensor_mm.
    # 24 mm * 6000 px / 36 mm = 4000 px exactly.
    payload = [{
        "SourceFile": "cam.JPG",
        "FocalLength": 24.0,
        "SensorWidth": 36.0,
        "ImageWidth": 6000,
    }]
    _mock_exiftool(monkeypatch, payload)
    ci = extract_intrinsics("cam.JPG", exiftool="exiftool")
    assert ci.focal_px == pytest.approx(24.0 * 6000 / 36.0)
    assert ci.focal_px == pytest.approx(4000.0)


def test_focal_mm_plus_focal_plane_resolution(monkeypatch):
    # Sensor width derived from FocalPlaneXResolution (px per unit) + unit code.
    # unit=4 (mm): sensor_mm = W_px / res = 6000 / 250 = 24 mm.
    # f_px = 12 mm * 6000 px / 24 mm = 3000 px.
    payload = [{
        "SourceFile": "cam.JPG",
        "FocalLength": 12.0,
        "FocalPlaneXResolution": 250.0,
        "FocalPlaneResolutionUnit": 4,
        "ExifImageWidth": 6000,
    }]
    _mock_exiftool(monkeypatch, payload)
    ci = extract_intrinsics("cam.JPG", exiftool="exiftool")
    assert ci.focal_px == pytest.approx(3000.0)


def test_focal_35mm_equivalent_fallback(monkeypatch):
    # Only the 35 mm-equivalent focal length: f_px = (f35 / 36) * W_px.
    # (24 / 36) * 5472 = 3648 px.
    payload = [{
        "SourceFile": "phone.JPG",
        "FocalLengthIn35mmFormat": 24.0,
        "ImageWidth": 5472,
    }]
    _mock_exiftool(monkeypatch, payload)
    ci = extract_intrinsics("phone.JPG", exiftool="exiftool")
    assert ci.focal_px == pytest.approx(24.0 / 36.0 * 5472)
    assert ci.focal_px == pytest.approx(3648.0)


def test_unparseable_focal_raises_valueerror(monkeypatch):
    # FocalLengthIn35mmFormat present but no image width -> nothing computable.
    payload = [{"SourceFile": "bad.JPG", "FocalLengthIn35mmFormat": 24.0}]
    _mock_exiftool(monkeypatch, payload)
    with pytest.raises(ValueError, match="focal length in pixels"):
        extract_intrinsics("bad.JPG", exiftool="exiftool")


def test_intrinsics_from_config():
    cfg = SimpleNamespace(camera=SimpleNamespace(focal_px=3713.0, hfov_deg=84.0))
    ci = intrinsics_from_config(cfg)
    assert isinstance(ci, CameraIntrinsics)
    assert ci.focal_px == pytest.approx(3713.0)
    assert ci.hfov_deg == pytest.approx(84.0)
    assert ci.principal_point is None


def test_intrinsics_from_focal_px():
    ci = intrinsics_from_focal_px(3713.0, hfov_deg=84.0, principal_point=(2736.0, 1824.0))
    assert ci.focal_px == pytest.approx(3713.0)
    assert ci.hfov_deg == pytest.approx(84.0)
    assert ci.principal_point == pytest.approx((2736.0, 1824.0))
    # hfov optional.
    assert intrinsics_from_focal_px(1000.0).hfov_deg is None


def test_exiftool_missing_raises(monkeypatch):
    monkeypatch.setattr(intrinsics, "find_exiftool", lambda: None)
    with pytest.raises(ExifToolNotFoundError, match="winget"):
        extract_intrinsics("DJI_0001.JPG")


def test_nonzero_returncode_surfaces_stderr(monkeypatch):
    _mock_exiftool(monkeypatch, returncode=1, stderr="Error: file not found")
    with pytest.raises(RuntimeError, match="file not found"):
        extract_intrinsics("missing.JPG", exiftool="exiftool")
