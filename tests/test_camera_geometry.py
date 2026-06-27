"""Oblique-tilt pose correction — fully offline, analytic geometry.

We forward-model a calibrated camera over a flat ground plane with a KNOWN pose
(camera centre, altitude, tilt, yaw), build the exact frame->ground homography it
induces, then check that `decompose_ground_homography` / `pose_from_homography`
recover the nadir (the drone's true map position) rather than the boresight (the
ground point under the optical axis), which an oblique camera offsets by
~altitude*tan(tilt). No network, no GPU, no fixtures.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronomy_loc.framework.schema import CameraIntrinsics  # noqa: E402
from dronomy_loc.localize.pipeline import (  # noqa: E402
    decompose_ground_homography, pose_from_homography,
)
from dronomy_loc.reference.geo import (  # noqa: E402
    GeoImage, haversine_m, lonlat_to_mercator, mercator_bbox_around,
)

LAT, LON = 43.521955, -5.624290   # Asturias prior (matches the rest of the suite)
F_PX = 3713.0                      # DJI Mavic 3E calibrated focal (px), per the plan


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


# world->cam for a nadir (straight-down) camera: optical axis = world -Z.
_R_NADIR = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)


def _project_ground_to_image(C, tilt_deg, yaw_deg, K):
    """Plane(E, N, 1) -> image px for a camera centred at C=(E,N,alt)."""
    R = _rx(math.radians(tilt_deg)) @ _R_NADIR @ _rz(math.radians(yaw_deg))
    t = -R @ np.asarray(C, float)
    return K @ np.column_stack([R[:, 0], R[:, 1], t])


def _K(w, h, f=F_PX):
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]], float)


# ── decompose_ground_homography: recover pose from a known camera ─────
@pytest.mark.parametrize("tilt", [0.0, 5.0, 15.0, 30.0])
def test_decompose_recovers_camera_pose(tilt):
    w, h = 1920, 1080
    K = _K(w, h)
    C = (40.0, -25.0, 90.0)                       # 90 m up, offset from any origin
    P = _project_ground_to_image(C, tilt, 33.0, K)
    G = np.linalg.inv(P)                           # image -> ground metres
    # Shift ground coords so the frame centre lands at the origin (the contract:
    # boresight at (0,0)); the recovered nadir is then C minus the boresight.
    bore = G @ np.array([w / 2.0, h / 2.0, 1.0])
    bore = bore[:2] / bore[2]
    shift = np.array([[1, 0, -bore[0]], [0, 1, -bore[1]], [0, 0, 1]], float)
    geom = decompose_ground_homography(shift @ G, K)
    assert geom is not None
    assert geom.altitude_m == pytest.approx(90.0, abs=0.5)
    assert geom.tilt_deg == pytest.approx(tilt, abs=0.5)
    assert geom.east_m == pytest.approx(C[0] - bore[0], abs=0.5)
    assert geom.north_m == pytest.approx(C[1] - bore[1], abs=0.5)


def test_decompose_rejects_singular_homography():
    K = _K(1920, 1080)
    assert decompose_ground_homography(np.zeros((3, 3)), K) is None


def test_decompose_rejects_grazing_tilt():
    # An 85 deg oblique is past the reliability cutoff -> None (fall back to boresight).
    w, h = 1920, 1080
    K = _K(w, h)
    P = _project_ground_to_image((10.0, 5.0, 90.0), 85.0, 0.0, K)
    G = np.linalg.inv(P)
    bore = G @ np.array([w / 2.0, h / 2.0, 1.0])
    bore = bore[:2] / bore[2]
    shift = np.array([[1, 0, -bore[0]], [0, 1, -bore[1]], [0, 0, 1]], float)
    assert decompose_ground_homography(shift @ G, K) is None


# ── pose_from_homography: end-to-end nadir vs boresight on a real tile ─
def _tile(span_m=300.0, pix=1024):
    bbox = mercator_bbox_around(LON, LAT, span_m)
    return GeoImage(image=np.zeros((pix, pix, 3), np.uint8), bbox=bbox)


def _homography_for(tile, C, tilt_deg, yaw_deg, frame_w, frame_h):
    """Build H (frame px -> tile px) for a camera with nadir at ground offset
    C=(E,N) metres from the tile centre, altitude C[2], given tilt/yaw."""
    minx, miny, maxx, maxy = tile.bbox
    cmx, cmy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    mpp_x, mpp_y = tile.meters_per_pixel
    cos_lat = math.cos(math.radians(LAT))
    # ground(E, N) relative to tile centre -> tile px (north-up, row 0 == top).
    M_gt = np.array([
        [1.0 / (cos_lat * mpp_x), 0.0, tile.width / 2.0],
        [0.0, -1.0 / (cos_lat * mpp_y), tile.height / 2.0],
        [0.0, 0.0, 1.0],
    ])
    K = _K(frame_w, frame_h)
    M_pi = _project_ground_to_image(C, tilt_deg, yaw_deg, K)   # ground -> frame px
    return M_gt @ np.linalg.inv(M_pi), cmx, cmy


def test_pose_tilt_correction_moves_to_nadir():
    tile = _tile()
    frame_w, frame_h = 1920, 1080
    alt, tilt, yaw = 90.0, 22.0, 18.0
    C = (35.0, -20.0, alt)
    H, cmx, cmy = _homography_for(tile, C, tilt, yaw, frame_w, frame_h)
    from dronomy_loc.reference.geo import mercator_to_lonlat
    cos_lat = math.cos(math.radians(LAT))
    nadir_lon, nadir_lat = mercator_to_lonlat(cmx + C[0] / cos_lat, cmy + C[1] / cos_lat)

    intr = CameraIntrinsics(focal_px=F_PX)
    naive = pose_from_homography(H, (frame_h, frame_w), tile)            # boresight
    fixed = pose_from_homography(H, (frame_h, frame_w), tile, intr)      # nadir

    naive_err = haversine_m(naive.lat, naive.lon, nadir_lat, nadir_lon)
    fixed_err = haversine_m(fixed.lat, fixed.lon, nadir_lat, nadir_lon)

    # The uncorrected boresight should be off by roughly altitude*tan(tilt).
    assert naive_err == pytest.approx(alt * math.tan(math.radians(tilt)), rel=0.15)
    assert naive_err > 25.0
    # The corrected pose lands on the nadir to sub-metre precision.
    assert fixed_err < 1.0
    assert fixed.tilt_corrected and not naive.tilt_corrected
    assert fixed.tilt_deg == pytest.approx(tilt, abs=0.5)
    assert fixed.altitude_m == pytest.approx(alt, abs=1.0)


def test_pose_without_intrinsics_is_unchanged():
    tile = _tile()
    H, _, _ = _homography_for(tile, (10.0, -5.0, 90.0), 0.0, 0.0, 1920, 1080)
    pose = pose_from_homography(H, (1080, 1920), tile)
    assert pose.tilt_deg is None and pose.altitude_m is None
    assert pose.tilt_corrected is False
    assert pose.lat is not None and pose.lon is not None


def test_nadir_camera_correction_is_negligible():
    # With zero tilt the nadir == boresight, so the correction must not move it.
    tile = _tile()
    intr = CameraIntrinsics(focal_px=F_PX)
    H, _, _ = _homography_for(tile, (15.0, 8.0, 100.0), 0.0, 27.0, 1920, 1080)
    naive = pose_from_homography(H, (1080, 1920), tile)
    fixed = pose_from_homography(H, (1080, 1920), tile, intr)
    assert haversine_m(naive.lat, naive.lon, fixed.lat, fixed.lon) < 0.5
    assert fixed.tilt_deg == pytest.approx(0.0, abs=0.5)
