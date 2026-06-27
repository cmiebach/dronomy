"""Export estimated + ground-truth tracks as GeoJSON (WGS84, [lon, lat] order).

Field crews and GIS tools (QGIS, geojson.io, Google Earth via import) read
GeoJSON directly, so this turns a list of `FrameScore`s into a map artifact: one
LineString for the estimated path (locked frames with a fix) and one for the
ground-truth path. Atomic write, like `validate.write_validation_csv`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..localize.validate import FrameScore


def _line_or_point(coords: list[list[float]], props: dict) -> dict | None:
    """A LineString needs >= 2 points; degrade to Point for one, skip for none."""
    if len(coords) >= 2:
        geom = {"type": "LineString", "coordinates": coords}
    elif len(coords) == 1:
        geom = {"type": "Point", "coordinates": coords[0]}
    else:
        return None
    return {"type": "Feature", "properties": props, "geometry": geom}


def tracks_geojson(rows: list[FrameScore], *, name: str = "") -> dict:
    """Build a FeatureCollection: estimated track + ground-truth track.
    Coordinates are [lon, lat] (GeoJSON order). Estimated track uses only frames
    with a fix; the GT track uses every row with finite truth."""
    est = [[r.est_lon, r.est_lat] for r in rows
           if r.est_lat is not None and r.est_lon is not None]
    gt = [[r.gt_lon, r.gt_lat] for r in rows
          if r.gt_lat == r.gt_lat and r.gt_lon == r.gt_lon]  # exclude NaN
    feats = []
    f_est = _line_or_point(est, {"track": "estimate", "name": name, "n": len(est)})
    f_gt = _line_or_point(gt, {"track": "ground_truth", "name": name, "n": len(gt)})
    if f_est:
        feats.append(f_est)
    if f_gt:
        feats.append(f_gt)
    return {"type": "FeatureCollection", "features": feats}


def write_geojson(rows: list[FrameScore], path: str | Path, *, name: str = "") -> Path:
    """Write the est + GT tracks to a .geojson file (atomic)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(tracks_geojson(rows, name=name), indent=2),
                   encoding="utf-8")
    os.replace(tmp, path)
    return path
