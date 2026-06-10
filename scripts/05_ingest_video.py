"""Small working piece #5: sharded, resumable, integrity-verified ingestion.

Usage:
    python scripts/05_ingest_video.py                    # uses config.yaml
    python scripts/05_ingest_video.py --every 2.0 --max 100
    python scripts/05_ingest_video.py --verify           # check an existing ingest
"""
import argparse

import _bootstrap  # noqa: F401
from dronomy_loc.config import load_config, resolve
from dronomy_loc.data import ingest


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=cfg.video.path)
    ap.add_argument("--out", default=getattr(cfg.frames, "ingest_dir", "data/ingest"))
    ap.add_argument("--every", type=float, default=cfg.frames.every_n_seconds,
                    help="seconds between sampled frames")
    ap.add_argument("--shard-seconds", type=float,
                    default=getattr(cfg.frames, "shard_seconds", 30.0),
                    help="length of one shard (resume/repair granularity)")
    ap.add_argument("--blur", choices=["sharpest", "off"],
                    default=getattr(cfg.frames, "blur_filter", "sharpest"),
                    help="'sharpest' = keep sharpest frame per window | 'off' = plain uniform")
    ap.add_argument("--min-blur-var", type=float,
                    default=getattr(cfg.frames, "min_blur_var", 0.0),
                    help="drop a window if even its sharpest frame scores below this")
    ap.add_argument("--max", type=int, default=getattr(cfg.frames, "max_frames", None),
                    help="cap on frames WRITTEN this run (run again to resume)")
    ap.add_argument("--force", action="store_true",
                    help="wipe a mismatched ingest dir and restart")
    ap.add_argument("--verify", action="store_true",
                    help="verify the existing ingest and exit")
    args = ap.parse_args()

    out = resolve(args.out)
    if args.verify:
        report = ingest.verify_ingest(out)
        print(f"Verified {report.n_checked} frames in {out}: "
              f"{'OK' if report.ok else f'{len(report.problems)} problem(s)'}")
        for p in report.problems:
            print("  !", p)
        if not report.ok:
            print("Damaged shards demoted to 'partial' -- re-run ingest to repair.")
        return

    video = resolve(args.video)
    print(f"Video: {video}")
    print(f"Ingesting ~1/{args.every:g}s into {args.shard_seconds:g}s shards, "
          f"blur_filter={args.blur} ...")
    res = ingest.ingest_video(
        video, out,
        every_n_seconds=args.every,
        shard_seconds=args.shard_seconds,
        blur_filter=args.blur,
        min_blur_var=args.min_blur_var,
        resize_long_edge=getattr(cfg.frames, "resize_long_edge", 1920),
        jpeg_quality=getattr(cfg.frames, "jpeg_quality", 95),
        max_frames=args.max,
        force=args.force,
    )
    state = "complete" if res.completed else "PARTIAL (re-run to resume)"
    print(f"Ingest {state}: {res.n_shards} shards | "
          f"{res.n_frames_written} written | {res.n_frames_skipped} skipped")
    print(f"Manifest: {res.manifest_path}")


if __name__ == "__main__":
    main()
