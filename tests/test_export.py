"""Export estimated + GT tracks to GeoJSON / KML. Offline, round-tripped."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronomy_loc.export import write_geojson, write_kml, tracks_geojson  # noqa: E402
from dronomy_loc.localize.validate import FrameScore  # noqa: E402


def _rows():
    out = []
    for f in range(3):
        out.append(FrameScore(
            frame=f, t_s=float(f),
            est_lat=43.5220 + f * 1e-4, est_lon=-5.6243 + f * 1e-4,
            gt_lat=43.5221 + f * 1e-4, gt_lon=-5.6244 + f * 1e-4,
            err_m=2.0, yaw_deg=10.0, n_inliers=40, locked=True, runtime_s=0.1))
    return out


def test_geojson_structure_and_coord_order(tmp_path):
    p = write_geojson(_rows(), tmp_path / "t.geojson", name="synthA")
    fc = json.loads(p.read_text())
    assert fc["type"] == "FeatureCollection" and len(fc["features"]) == 2
    est = next(f for f in fc["features"] if f["properties"]["track"] == "estimate")
    gt = next(f for f in fc["features"] if f["properties"]["track"] == "ground_truth")
    assert est["geometry"]["type"] == "LineString"
    assert len(est["geometry"]["coordinates"]) == 3
    # GeoJSON order is [lon, lat]
    assert est["geometry"]["coordinates"][0] == [-5.6243, 43.5220]
    assert gt["geometry"]["coordinates"][0] == [-5.6244, 43.5221]


def test_geojson_skips_missing_estimate_track():
    rows = [FrameScore(0, 0.0, None, None, 43.5, -5.6, None, None, 0, False, 0.1),
            FrameScore(1, 1.0, None, None, 43.5, -5.6, None, None, 0, False, 0.1)]
    fc = tracks_geojson(rows)
    tracks = {f["properties"]["track"] for f in fc["features"]}
    assert tracks == {"ground_truth"}   # no estimate feature when nothing locked


def test_kml_has_both_placemarks(tmp_path):
    p = write_kml(_rows(), tmp_path / "t.kml", name="synthA")
    txt = p.read_text()
    assert txt.startswith("<?xml") and "<kml" in txt and txt.rstrip().endswith("</kml>")
    assert txt.count("<LineString>") == 2
    assert "Estimate" in txt and "Ground truth" in txt
    assert "-5.62430000,43.52200000,0" in txt   # lon,lat,alt order
