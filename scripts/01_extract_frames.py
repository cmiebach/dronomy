"""Small working piece #1: read the drone video and extract sampled frames.

Usage:
    python scripts/01_extract_frames.py                 # uses config.yaml
    python scripts/01_extract_frames.py --every 2.0 --max 30 --probe
"""
import argparse

import _bootstrap  # noqa: F401
from dronomy_loc.config import load_config, resolve
from dronomy_loc.data import frames


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=float, default=cfg.frames.every_n_seconds,
                    help="seconds between sampled frames")
    ap.add_argument("--max", type=int, default=cfg.frames.max_frames)
    ap.add_argument("--blur", choices=["sharpest", "off"],
                    default=getattr(cfg.frames, "blur_filter", "sharpest"),
                    help="'sharpest' = keep sharpest frame per window | 'off' = plain uniform")
    ap.add_argument("--min-blur-var", type=float,
                    default=getattr(cfg.frames, "min_blur_var", 0.0),
                    help="drop a window if even its sharpest frame scores below this")
    ap.add_argument("--probe", action="store_true", help="print video metadata and exit")
    args = ap.parse_args()

    video = resolve(cfg.video.path)
    print(f"Video: {video}")
    print("Metadata:", frames.probe(video))
    if args.probe:
        return

    out = resolve(cfg.frames.out_dir)
    print(f"Extracting ~1/{args.every:g}s, blur_filter={args.blur} ...")
    paths = frames.extract_frames(
        video, out,
        every_n_seconds=args.every,
        max_frames=args.max,
        resize_long_edge=cfg.frames.resize_long_edge,
        jpeg_quality=cfg.frames.jpeg_quality,
        blur_filter=args.blur,
        min_blur_var=args.min_blur_var,
    )
    print(f"Wrote {len(paths)} frames to {out}")
    if paths:
        print("First:", paths[0].name, "| Last:", paths[-1].name)
        print(f"Manifest: {out / 'frames.csv'}")


if __name__ == "__main__":
    main()
