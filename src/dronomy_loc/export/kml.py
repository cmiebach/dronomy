"""Export estimated + ground-truth tracks as KML (opens in Google Earth).

KML coordinates are "lon,lat[,alt]" tuples, space-separated, inside a LineString.
We emit two Placemarks (estimate, ground truth). Kept dependency-free (string
templating) and ASCII-only so it works on any console; atomic write.
"""
from __future__ import annotations

import os
from pathlib import Path

from ..localize.validate import FrameScore

_HEADER = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>')
_FOOTER = "</Document></kml>"


def _placemark(name: str, coords: list[tuple[float, float]], color: str) -> str:
    if len(coords) < 1:
        return ""
    pts = " ".join(f"{lon:.8f},{lat:.8f},0" for lon, lat in coords)
    if len(coords) == 1:
        geom = f"<Point><coordinates>{pts}</coordinates></Point>"
    else:
        geom = ("<LineString><tessellate>1</tessellate>"
                f"<coordinates>{pts}</coordinates></LineString>")
    style = (f'<Style><LineStyle><color>{color}</color><width>3</width>'
             "</LineStyle></Style>")
    return f"<Placemark><name>{name}</name>{style}{geom}</Placemark>"


def tracks_kml(rows: list[FrameScore], *, name: str = "") -> str:
    """KML document string with an estimated and a ground-truth track."""
    est = [(r.est_lon, r.est_lat) for r in rows
           if r.est_lat is not None and r.est_lon is not None]
    gt = [(r.gt_lon, r.gt_lat) for r in rows
          if r.gt_lat == r.gt_lat and r.gt_lon == r.gt_lon]
    label = f" ({name})" if name else ""
    body = (_placemark(f"Estimate{label}", est, "ff0000ff")     # KML aabbggrr: red
            + _placemark(f"Ground truth{label}", gt, "ff00ff00"))  # green
    return _HEADER + body + _FOOTER


def write_kml(rows: list[FrameScore], path: str | Path, *, name: str = "") -> Path:
    """Write the est + GT tracks to a .kml file (atomic)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(tracks_kml(rows, name=name), encoding="utf-8")
    os.replace(tmp, path)
    return path
