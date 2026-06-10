"""Small working piece #6: extract the embedded GPS track (GROUND TRUTH only).

The DJI Mavic 3 Enterprise logs one GPS sample per video frame in the `djmd`
metadata stream; exiftool decodes it. The resulting CSV is used ONLY to score
the localizer's output — it is never fed into the localization itself.

Usage:
    python scripts/06_extract_gps_track.py                  # uses config.yaml
    python scripts/06_extract_gps_track.py --video v.mp4 --out data/gps_track.csv
"""
import argparse
import sys

import _bootstrap  # noqa: F401
from dronomy_loc.config import load_config, resolve
from dronomy_loc.data.telemetry import ExifToolNotFoundError, extract_gps_track


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=cfg.video.path)
    ap.add_argument("--out",
                    default=getattr(cfg.video, "gps_track_csv", "data/gps_track.csv"))
    ap.add_argument("--exiftool", default=None,
                    help="path to exiftool.exe (default: auto-detect)")
    args = ap.parse_args()

    video = resolve(args.video)
    out = resolve(args.out)
    print(f"Video: {video}")
    try:
        fixes = extract_gps_track(video, out_csv=out, exiftool=args.exiftool)
    except ExifToolNotFoundError as e:
        print(e)
        sys.exit(1)

    if not fixes:
        print("No valid GPS fixes found in the video metadata.")
        sys.exit(1)

    lats = [f.lat for f in fixes]
    lons = [f.lon for f in fixes]
    print(f"Extracted {len(fixes)} GPS fixes -> {out}")
    print(f"First: frame {fixes[0].frame}  lat={fixes[0].lat:.6f} lon={fixes[0].lon:.6f}")
    print(f"Last:  frame {fixes[-1].frame}  lat={fixes[-1].lat:.6f} lon={fixes[-1].lon:.6f}")
    print(f"BBox:  lat [{min(lats):.6f}, {max(lats):.6f}]  "
          f"lon [{min(lons):.6f}, {max(lons):.6f}]")
    print("Reminder: this track is GROUND TRUTH for scoring only -- "
          "never an input to the localizer.")


if __name__ == "__main__":
    main()
