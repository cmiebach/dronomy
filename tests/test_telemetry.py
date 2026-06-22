"""Telemetry (GPS ground-truth) extraction — all offline, exiftool is mocked."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronomy_loc.data import telemetry  # noqa: E402
from dronomy_loc.data.telemetry import (  # noqa: E402
    ExifToolNotFoundError, GPSFix, extract_gps_track, gt_for_frame,
    load_track_csv, write_track_csv,
)


def _mock_exiftool(monkeypatch, payload=None, returncode=0, stderr=""):
    stdout = json.dumps(payload) if payload is not None else ""

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            self.pid = 4242
            self.returncode = None

        def communicate(self, timeout=None):
            self.returncode = returncode
            return stdout, stderr

    monkeypatch.setattr(telemetry.subprocess, "Popen", FakePopen)


# Doc groups deliberately out of order, with Doc10 to catch lexicographic sorting
# ("Doc10" < "Doc2" as strings). Doc3 is the classic invalid (0,0) fix.
G3_PAYLOAD = [{
    "SourceFile": "v.mp4",
    "Doc10:GPSLatitude": 43.52200, "Doc10:GPSLongitude": -5.62433,
    "Doc10:GPSAltitude": 122.0, "Doc10:SampleTime": 0.300,
    "Doc1:GPSLatitude": 43.52196, "Doc1:GPSLongitude": -5.62429,
    "Doc1:GPSAltitude": 120.5, "Doc1:SampleTime": 0.0,
    "Doc3:GPSLatitude": 0.0, "Doc3:GPSLongitude": 0.0,
    "Doc4:GPSLatitude": 43.52199, "Doc4:GPSLongitude": -5.62432,
    "Doc2:GPSLatitude": 43.52197, "Doc2:GPSLongitude": -5.62430,
    "Doc2:GPSAltitude": 121.0, "Doc2:SampleTime": 0.033,
}]


def test_parse_g3_doc_groups(monkeypatch):
    _mock_exiftool(monkeypatch, G3_PAYLOAD)
    fixes = extract_gps_track("v.mp4", exiftool="exiftool")

    # 5 docs, the (0,0) one dropped; frame = DocN - 1 (doc-derived, NOT reindexed
    # after filtering), sorted numerically by N.
    assert [f.frame for f in fixes] == [0, 1, 3, 9]
    assert fixes[0].lat == pytest.approx(43.52196)
    assert fixes[0].lon == pytest.approx(-5.62429)
    assert fixes[0].alt_m == pytest.approx(120.5)
    assert fixes[0].t_s == pytest.approx(0.0)
    assert fixes[1].t_s == pytest.approx(0.033)
    assert fixes[2].t_s is None        # Doc4 has no SampleTime
    assert fixes[2].alt_m is None      # ... nor altitude
    assert fixes[3].lat == pytest.approx(43.52200)


def test_parse_flat_single_fix_fallback(monkeypatch):
    payload = [{"SourceFile": "v.mp4",
                "Composite:GPSLatitude": 43.5, "Composite:GPSLongitude": -5.6,
                "Composite:GPSAltitude": 100.0}]
    _mock_exiftool(monkeypatch, payload)
    fixes = extract_gps_track("v.mp4", exiftool="exiftool")
    assert len(fixes) == 1
    assert fixes[0] == GPSFix(frame=0, t_s=None, lat=43.5, lon=-5.6, alt_m=100.0)


def test_csv_roundtrip(tmp_path):
    fixes = [
        GPSFix(frame=0, t_s=0.0, lat=43.52196, lon=-5.62429, alt_m=120.5),
        GPSFix(frame=1, t_s=None, lat=43.52197, lon=-5.62430, alt_m=None),
        GPSFix(frame=9, t_s=0.300, lat=43.52200, lon=-5.62433, alt_m=122.0),
    ]
    out = tmp_path / "sub" / "gps_track.csv"   # exercises parent-dir creation
    write_track_csv(fixes, out)
    loaded = load_track_csv(out)
    assert len(loaded) == len(fixes)
    for a, b in zip(loaded, fixes):
        assert a.frame == b.frame
        assert a.lat == pytest.approx(b.lat)
        assert a.lon == pytest.approx(b.lon)
        if b.t_s is None:
            assert a.t_s is None
        else:
            assert a.t_s == pytest.approx(b.t_s)
        if b.alt_m is None:
            assert a.alt_m is None
        else:
            assert a.alt_m == pytest.approx(b.alt_m)


def test_gt_for_frame_exact_and_nearest():
    fixes = [GPSFix(frame=f, t_s=None, lat=43.0 + f, lon=-5.0, alt_m=None)
             for f in (0, 1, 3, 9)]
    assert gt_for_frame(fixes, 3).frame == 3     # exact
    assert gt_for_frame(fixes, 7).frame == 9     # nearest (|7-9| < |7-3|)
    assert gt_for_frame(fixes, 2).frame in (1, 3)  # tie: either neighbour is fine
    with pytest.raises(ValueError):
        gt_for_frame([], 0)


def test_exiftool_missing_raises_with_install_hint(monkeypatch):
    monkeypatch.setattr(telemetry, "find_exiftool", lambda: None)
    with pytest.raises(ExifToolNotFoundError, match="winget"):
        extract_gps_track("v.mp4")


def test_subprocess_failure_surfaces_stderr(monkeypatch):
    _mock_exiftool(monkeypatch, returncode=1, stderr="Error: bad atom in v.mp4")
    with pytest.raises(RuntimeError, match="bad atom"):
        extract_gps_track("v.mp4", exiftool="exiftool")


def test_sampletime_string_parses(monkeypatch):
    payload = [{"SourceFile": "v.mp4",
                "Doc1:GPSLatitude": 43.5, "Doc1:GPSLongitude": -5.6,
                "Doc1:SampleTime": "1.234 s"}]
    _mock_exiftool(monkeypatch, payload)
    fixes = extract_gps_track("v.mp4", exiftool="exiftool")
    assert fixes[0].t_s == pytest.approx(1.234)


def test_flat_fallback_keeps_element_positions(monkeypatch):
    # One sample per array element; the GPS-less element must still consume a
    # frame slot or every later frame index silently shifts (GT misalignment).
    payload = [
        {"SourceFile": "v.mp4"},  # container object: no telemetry tags, no slot
        {"GPSLatitude": 43.1, "GPSLongitude": -5.1},
        {"GPSLatitude": 43.2, "GPSLongitude": -5.2},
        {"SampleTime": 0.066},    # sample element with a lost fix: slot, no fix
        {"GPSLatitude": 43.4, "GPSLongitude": -5.4},
    ]
    _mock_exiftool(monkeypatch, payload)
    fixes = extract_gps_track("v.mp4", exiftool="exiftool")
    assert [f.frame for f in fixes] == [0, 1, 3]   # NOT [0, 1, 2]
    assert fixes[2].lat == pytest.approx(43.4)


def test_timeout_kills_process_tree_and_raises(monkeypatch):
    kills = []

    class HangingPopen:
        def __init__(self, cmd, **kwargs):
            self.pid = 4242
            self.returncode = None
            self._timed_out = False

        def communicate(self, timeout=None):
            if not self._timed_out and timeout is not None:
                self._timed_out = True
                raise subprocess.TimeoutExpired(cmd="exiftool", timeout=timeout)
            return "", ""

        def kill(self):
            kills.append("kill")

    monkeypatch.setattr(telemetry.subprocess, "Popen", HangingPopen)
    monkeypatch.setattr(telemetry.subprocess, "run",
                        lambda cmd, **kw: kills.append(tuple(cmd[:1])))
    with pytest.raises(RuntimeError, match="timed out"):
        extract_gps_track("v.mp4", exiftool="exiftool", timeout_s=1)
    assert kills  # the tree-kill path ran (taskkill on Windows, kill() elsewhere)
