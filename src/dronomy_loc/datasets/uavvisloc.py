"""UAV-VisLoc adapter: a public multi-region benchmark -> one `Scenario` per region.

This adapter is the generality proof for the framework: the SAME localizer that
runs on the provided Asturias flight runs unchanged on an external, multi-region
public benchmark, with zero model-side changes. Each UAV-VisLoc region becomes an
independent `Scenario` (its own reference map + its own drone images), so the
runner/report group results per region/terrain for free.

ON-DISK LAYOUT (what this adapter expects under the dataset ROOT)::

    <root>/
        satellite_coordinates_range.csv      # top-level: each map's geo extent
        01/
            drone/  00001.JPG 00002.JPG ...  # drone query images (.JPG or .jpg)
            satellite01.tif (or .png)        # one georeferenced satellite map
            01.csv                           # per-image GT (lat/lon[/height/heading])
        02/ ...

GT join: a per-region CSV (`<region>/<region>.csv`, else any other `*.csv` in the
region dir, else a top-level CSV) maps each drone filename -> latitude/longitude
(+ optional height, heading). The satellite map's geographic extent comes from the
top-level `satellite_coordinates_range.csv`, keyed by the map filename.

COLUMN NAMES VARY across UAV-VisLoc releases, so every lookup is tolerant:
headers are lowercased/stripped and matched against alias lists (see `_ALIASES`
and `_RANGE_ALIASES`). A region whose files or GT are missing is SKIPPED with a
clear printed note rather than aborting the whole dataset — a partial download
still yields the scenarios it can.

Pure-Python georeferencing (no rasterio): the lon/lat corners from the range CSV
are converted to EPSG:3857 via `lonlat_to_mercator`, wrapped in a `GeoImage`, and
served through `make_world_fetch` + `TileCache` exactly like the one-world-tile
optimization the validator already uses (crop-locally, zero per-call network). If
the range bbox is absent AND `rasterio` is importable, the .tif geotransform is
read via a lazy import as a fallback; otherwise a clear error names the missing CSV.

GROUND-TRUTH RULE: each `Sample.gt` is for scoring only and is NEVER fed to the
localizer — UAV-VisLoc is treated as telemetry-free, same as the provided flight.
"""
from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from ..config import resolve
from ..data.telemetry import GPSFix
from ..framework.schema import FetchTile, Sample, Scenario
from ..localize.search import TileCache
from ..localize.validate import make_world_fetch
from ..reference.geo import GeoImage, lonlat_to_mercator
from .base import Dataset

# Default location if neither an arg nor a cfg.datasets entry points elsewhere.
_DEFAULT_ROOT = "data/datasets/uav-visloc"

# Drone-image extensions, lowercased (matched case-insensitively below).
_IMG_EXTS = (".jpg", ".jpeg", ".png")
# Satellite-map basename prefixes / extensions to recognise.
_SAT_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

# GT-CSV column aliases (header lowercased+stripped -> our field). First match wins.
_ALIASES = {
    "lat": ("lat", "latitude", "gps_lat", "drone_lat", "lat_deg"),
    "lon": ("lon", "lng", "long", "longitude", "gps_lon", "drone_lon", "lon_deg"),
    "filename": ("filename", "file", "image", "image_name", "name", "img", "drone"),
    "height": ("height", "alt", "altitude", "h", "rel_alt", "relativealtitude"),
    "heading": ("heading", "yaw", "course", "drone_heading", "gimbalyaw"),
}

# Range-CSV aliases: each map's geographic extent (left-top / right-bottom corners
# OR generic min/max). LT = left-top (min lon, max lat); RB = right-bottom.
_RANGE_ALIASES = {
    "mapname": ("mapname", "map", "filename", "file", "image", "name", "satellite"),
    "lt_lat": ("lt_lat_map", "lat_lt", "lt_lat", "tl_lat", "lat_max", "maxlat",
               "max_lat", "north", "top"),
    "lt_lon": ("lt_lon_map", "lon_lt", "lt_lon", "tl_lon", "lon_min", "minlon",
               "min_lon", "west", "left"),
    "rb_lat": ("rb_lat_map", "lat_rb", "rb_lat", "br_lat", "lat_min", "minlat",
               "min_lat", "south", "bottom"),
    "rb_lon": ("rb_lon_map", "lon_rb", "rb_lon", "br_lon", "lon_max", "maxlon",
               "max_lon", "east", "right"),
}


def _norm(s: str) -> str:
    """Header normaliser: lowercase, strip, drop spaces so 'LT_lat map' matches."""
    return s.strip().lower().replace(" ", "").replace("-", "_")


def _pick(row: dict, aliases: tuple[str, ...]) -> str | None:
    """First value in `row` whose normalised key is in `aliases` (None if none)."""
    for key, val in row.items():
        if key is not None and _norm(key) in aliases:
            v = (val or "").strip() if isinstance(val, str) else val
            if v not in (None, ""):
                return v
    return None


def _to_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _read_csv_rows(path: Path) -> list[dict]:
    """Read a CSV into a list of {raw_header: value} dicts (utf-8-sig eats a BOM)."""
    with open(path, "r", newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _imdecode(path: Path, flag: int) -> np.ndarray | None:
    """Windows non-ASCII-safe decode (np.fromfile + cv2.imdecode). None on failure."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, flag)
    return img  # None if cv2 could not decode


def _stem_key(name: str) -> str:
    """Match key for a filename: lowercased basename without extension."""
    return Path(name).stem.lower()


def _digits(s: str) -> str:
    """Digit run in a name, e.g. 'satellite03' -> '03', '03.tif' -> '03'."""
    return "".join(ch for ch in s if ch.isdigit())


def _find_range_csv(root: Path) -> Path | None:
    """Locate the satellite-extent CSV at the dataset root. The real UAV-VisLoc
    release ships it with a stray space ('satellite_ coordinates_range.csv'), so
    match tolerantly on normalised name rather than an exact string."""
    exact = root / "satellite_coordinates_range.csv"
    if exact.is_file():
        return exact
    for p in sorted(root.glob("*.csv")):
        n = _norm(p.name)
        if "range" in n and ("coordinate" in n or "satellite" in n):
            return p
    return None


# ── GT and range CSV parsing ──────────────────────────────────────────


def _load_gt_by_filename(
    csv_path: Path,
) -> dict[str, tuple[float, float, float | None, float | None]]:
    """Map drone-filename-stem -> (lat, lon, height_or_None, heading_or_None).
    Rows missing lat/lon are skipped. The filename column is matched tolerantly.
    (Tuple, not GPSFix: frame is unknown until the sample loop assigns it, and
    heading lives in Sample.meta, not on the GPSFix.)"""
    gt: dict[str, tuple[float, float, float | None, float | None]] = {}
    for row in _read_csv_rows(csv_path):
        fname = _pick(row, _ALIASES["filename"])
        lat = _to_float(_pick(row, _ALIASES["lat"]))
        lon = _to_float(_pick(row, _ALIASES["lon"]))
        if fname is None or lat is None or lon is None:
            continue
        alt = _to_float(_pick(row, _ALIASES["height"]))
        heading = _to_float(_pick(row, _ALIASES["heading"]))
        gt[_stem_key(str(fname))] = (lat, lon, alt, heading)
    return gt


def _load_ranges(csv_path: Path) -> dict[str, tuple[float, float, float, float]]:
    """Map satellite-map-stem -> (min_lon, min_lat, max_lon, max_lat) WGS84.
    Tolerant of LT/RB corner columns or generic min/max columns."""
    ranges: dict[str, tuple[float, float, float, float]] = {}
    for row in _read_csv_rows(csv_path):
        mapname = _pick(row, _RANGE_ALIASES["mapname"])
        lt_lat = _to_float(_pick(row, _RANGE_ALIASES["lt_lat"]))
        lt_lon = _to_float(_pick(row, _RANGE_ALIASES["lt_lon"]))
        rb_lat = _to_float(_pick(row, _RANGE_ALIASES["rb_lat"]))
        rb_lon = _to_float(_pick(row, _RANGE_ALIASES["rb_lon"]))
        if None in (mapname, lt_lat, lt_lon, rb_lat, rb_lon):
            continue
        min_lon, max_lon = sorted((lt_lon, rb_lon))   # corner order varies; sort
        min_lat, max_lat = sorted((lt_lat, rb_lat))
        bbox = (min_lon, min_lat, max_lon, max_lat)
        ranges[_stem_key(str(mapname))] = bbox
        # Also key by digit run: the range CSV says '03.tif' while the map file
        # is 'satellite03.tif' -> link them by the shared region number '03'.
        d = _digits(str(mapname))
        if d:
            ranges.setdefault(d, bbox)
    return ranges


def _range_lookup(ranges: dict, sat_path: Path, region: str):
    """Find a map's bbox tolerant of map-name vs range-name mismatches: try the
    satellite file stem, the region id, and their digit runs in turn."""
    for key in (_stem_key(sat_path.name), region.lower(),
                _digits(sat_path.stem), _digits(region)):
        if key and key in ranges:
            return ranges[key]
    return None


def _bbox_from_rasterio(tif_path: Path) -> tuple[float, float, float, float] | None:
    """Fallback: read a GeoTIFF's WGS84 bounds via a LAZY rasterio import. Returns
    (min_lon, min_lat, max_lon, max_lat) or None if rasterio/CRS is unavailable."""
    try:
        import rasterio                       # lazy: optional dependency
        from rasterio.warp import transform_bounds
    except Exception:
        return None
    try:
        with rasterio.open(str(tif_path)) as ds:
            if ds.crs is None:
                return None
            left, bottom, right, top = transform_bounds(
                ds.crs, "EPSG:4326", *ds.bounds)
        return (min(left, right), min(bottom, top),
                max(left, right), max(bottom, top))
    except Exception:
        return None


# ── region discovery ──────────────────────────────────────────────────


def _find_drone_dir(region_dir: Path) -> Path | None:
    """The folder holding the query images: prefer 'drone', else the region dir
    itself if it directly contains images."""
    drone = region_dir / "drone"
    if drone.is_dir():
        return drone
    if any(p.suffix.lower() in _IMG_EXTS for p in region_dir.iterdir() if p.is_file()):
        return region_dir
    return None


def _list_drone_images(drone_dir: Path) -> list[Path]:
    return sorted(p for p in drone_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in _IMG_EXTS)


def _find_satellite(region_dir: Path) -> Path | None:
    """The region's georeferenced map: a file whose name starts with 'satellite'
    (any recognised raster ext), else the single largest non-drone raster."""
    cands = [p for p in region_dir.iterdir()
             if p.is_file() and p.suffix.lower() in _SAT_EXTS]
    sat_named = [p for p in cands if p.name.lower().startswith("satellite")]
    pool = sat_named or cands
    if not pool:
        return None
    return max(pool, key=lambda p: p.stat().st_size)


def _find_gt_csv(region_dir: Path, region: str, root: Path) -> Path | None:
    """The GT CSV for a region: '<region>/<region>.csv', else any other region
    CSV (excluding the range CSV), else a top-level '<region>.csv'."""
    named = region_dir / f"{region}.csv"
    if named.is_file():
        return named
    for p in sorted(region_dir.glob("*.csv")):
        if p.name.lower() != "satellite_coordinates_range.csv":
            return p
    top = root / f"{region}.csv"
    return top if top.is_file() else None


def _list_region_dirs(root: Path) -> list[Path]:
    """Region subfolders: any directory that holds drone images. Sorted by name so
    scenarios are deterministic."""
    return sorted((p for p in root.iterdir()
                   if p.is_dir() and _find_drone_dir(p) is not None),
                  key=lambda p: p.name)


# ── Scenario construction ─────────────────────────────────────────────


def _make_samples(
    images: list[Path],
    gt_by_name: dict[str, tuple[float, float, float | None, float | None]],
    region: str,
):
    """Build a zero-arg factory returning a FRESH lazy/streaming Sample iterator.
    Reads each image only when the consumer pulls it (Scenario can be replayed).
    `frame_id` is the running 0-based index over the SORTED image list and is the
    GT join key, so it stays stable across replays."""

    def factory() -> Iterator[Sample]:
        for idx, img_path in enumerate(images):
            img = _imdecode(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"uavvisloc: region {region}: could not decode "
                      f"{img_path.name}, skipping")
                continue
            raw = gt_by_name.get(_stem_key(img_path.name))
            heading = None
            gt = None
            if raw is not None:
                lat, lon, alt, heading = raw
                gt = GPSFix(frame=idx, t_s=None, lat=lat, lon=lon, alt_m=alt)
            yield Sample(
                frame_id=idx,
                image_bgr=img,
                t_s=None,
                gt=gt,
                intrinsics=None,
                meta={"filename": img_path.name, "heading": heading,
                      "region": region},
            )

    return factory


def _load_satellite_rgb(sat_path: Path, max_edge: int | None) -> np.ndarray | None:
    """Decode a satellite map to RGB, downscaled so the longest edge is <=
    `max_edge`. Real UAV-VisLoc maps are huge (e.g. 35092x24308 ~ 2.6 GB decoded);
    the crops we serve are only a few hundred px, so a capped raster keeps memory
    sane WITHOUT changing georeferencing (same ground extent, fewer pixels). cv2
    for normal-sized files; PIL for very large ones — cv2.imdecode overflows
    (signed-int size assert) on ~2 GB+ buffers, so we do NOT hand it those."""
    if sat_path.stat().st_size <= 1_200_000_000:
        try:
            bgr = _imdecode(sat_path, cv2.IMREAD_COLOR)
        except cv2.error:
            bgr = None
        if bgr is not None:
            h, w = bgr.shape[:2]
            if max_edge and max(h, w) > max_edge:
                s = max_edge / max(h, w)
                bgr = cv2.resize(bgr, (round(w * s), round(h * s)),
                                 interpolation=cv2.INTER_AREA)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    try:                                    # PIL path for very large rasters
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None       # lift the decompression-bomb guard
        im = Image.open(str(sat_path))
        if max_edge:
            im.draft("RGB", (max_edge, max_edge))   # decode at reduced scale when possible
        im = im.convert("RGB")
        if max_edge and max(im.size) > max_edge:
            s = max_edge / max(im.size)
            im = im.resize((round(im.size[0] * s), round(im.size[1] * s)))
        return np.asarray(im)
    except Exception:
        return None


def _make_fetch_tile(sat_path: Path,
                     bbox_wgs84: tuple[float, float, float, float],
                     max_sat_edge: int | None = 8192) -> FetchTile:
    """Load the satellite map pixels (RGB, capped) + its WGS84 bbox -> one
    georeferenced `GeoImage` -> `TileCache(make_world_fetch(world))`
    (crop-locally, no network)."""
    rgb = _load_satellite_rgb(sat_path, max_sat_edge)
    if rgb is None:
        raise FileNotFoundError(f"could not decode satellite map: {sat_path}")
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    minx, miny = lonlat_to_mercator(min_lon, min_lat)
    maxx, maxy = lonlat_to_mercator(max_lon, max_lat)
    world = GeoImage(image=rgb, bbox=(minx, miny, maxx, maxy))
    return TileCache(make_world_fetch(world))


def _resolve_root(cfg, root) -> Path:
    """root precedence: explicit arg -> the cfg.datasets entry whose type is
    'uavvisloc' -> the default path. Resolved config-relative against repo root."""
    if root is not None:
        return Path(root)
    path_str = _DEFAULT_ROOT
    entries = getattr(cfg, "datasets", None) if cfg is not None else None
    if entries:
        for entry in entries:
            if getattr(entry, "type", None) == "uavvisloc":
                path_str = getattr(entry, "path", None) or _DEFAULT_ROOT
                break
    return resolve(path_str)


class UAVVisLocDataset(Dataset):
    """UAV-VisLoc public benchmark -> one `Scenario` per region directory."""

    def __init__(self, cfg=None, root: str | Path | None = None):
        self.cfg = cfg
        self.root = _resolve_root(cfg, root)

    def scenarios(self) -> list[Scenario]:
        """One Scenario per region that has drone images AND resolvable GT + a
        georeferenced satellite map. Regions missing any piece are skipped with a
        printed note (a partial download still yields what it can)."""
        root = self.root
        if not root.is_dir():
            print(f"uavvisloc: dataset root not found: {root} "
                  f"(a real run needs the UAV-VisLoc sample under this path)")
            return []

        range_csv = _find_range_csv(root)
        ranges = _load_ranges(range_csv) if range_csv else {}

        out: list[Scenario] = []
        for region_dir in _list_region_dirs(root):
            region = region_dir.name
            drone_dir = _find_drone_dir(region_dir)
            images = _list_drone_images(drone_dir) if drone_dir else []
            if not images:
                print(f"uavvisloc: region {region}: no drone images, skipping")
                continue

            gt_csv = _find_gt_csv(region_dir, region, root)
            if gt_csv is None:
                print(f"uavvisloc: region {region}: no GT CSV found, skipping")
                continue
            gt_by_name = _load_gt_by_filename(gt_csv)

            sat_path = _find_satellite(region_dir)
            if sat_path is None:
                print(f"uavvisloc: region {region}: no satellite map, skipping")
                continue

            bbox = _range_lookup(ranges, sat_path, region)
            if bbox is None:
                bbox = _bbox_from_rasterio(sat_path)
            if bbox is None:
                where = range_csv.name if range_csv else "no range CSV at root"
                print(f"uavvisloc: region {region}: no geo extent for "
                      f"{sat_path.name} (missing from {where} and no "
                      f"rasterio geotransform), skipping")
                continue

            try:
                fetch_tile = _make_fetch_tile(sat_path, bbox)
            except FileNotFoundError as e:
                print(f"uavvisloc: region {region}: {e}, skipping")
                continue

            min_lon, min_lat, max_lon, max_lat = bbox
            center = ((min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0)  # (lat, lon)

            out.append(Scenario(
                name=f"uavvisloc-{region}",
                terrain="unknown",   # do NOT hardcode a wrong terrain per region
                fetch_tile=fetch_tile,
                sample_iter=_make_samples(images, gt_by_name, region),
                prior=center,
                intrinsics=None,
                meta={"dataset": "uavvisloc", "region": region,
                      "n_drone": len(images)},
            ))
        return out
