"""Small working piece #2: fetch a georeferenced satellite tile for the area.

Usage:
    python scripts/02_fetch_reference.py                  # uses config.yaml
    python scripts/02_fetch_reference.py --provider ign --span 1500 --pixels 4096
"""
import argparse

import _bootstrap  # noqa: F401

from dronomy_loc.config import load_config, resolve
from dronomy_loc.reference import get_provider, save_reference


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=cfg.reference.provider, choices=["ign", "gee"])
    ap.add_argument("--lat", type=float, default=cfg.video.rough_lat)
    ap.add_argument("--lon", type=float, default=cfg.video.rough_lon)
    ap.add_argument("--span", type=float, default=cfg.reference.span_meters)
    ap.add_argument("--pixels", type=int, default=cfg.reference.pixels)
    args = ap.parse_args()

    print(f"Fetching {args.provider} tile @ ({args.lat}, {args.lon}), "
          f"{args.span} m, {args.pixels}px ...")
    provider = get_provider(args.provider, cfg)
    geo = provider.fetch(args.lat, args.lon, args.span, args.pixels)

    out_dir = resolve(cfg.reference.out_dir)
    img_path = save_reference(geo, out_dir, args.provider)

    mpp = geo.meters_per_pixel
    print(f"Saved {img_path}  ({geo.width}x{geo.height}, ~{mpp[0]:.3f} m/px)")
    # Round-trip sanity check: center pixel should map back to ~(lat, lon).
    clon, clat = geo.pixel_to_lonlat(geo.width / 2, geo.height / 2)
    print(f"Center pixel -> ({clat:.6f}, {clon:.6f})  [requested ({args.lat}, {args.lon})]")


if __name__ == "__main__":
    main()
