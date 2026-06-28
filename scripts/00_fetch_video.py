"""Step 0 — data ingestion: pull the drone video onto this machine.

This is the FIRST step of the end-to-end run. The flight video is ~3.7 GB and
lives off-repo (a Dropbox share), so a fresh clone has no footage to localize.
This downloads it (resumable + size-verified) to the path the rest of the
pipeline expects (`config.video.path`). Re-running is a no-op once complete.

Usage:
    python scripts/00_fetch_video.py                       # uses config.yaml
    python scripts/00_fetch_video.py --url <URL> --out <path>
    python scripts/00_fetch_video.py --force               # ignore an existing file
"""
import argparse
import sys

import _bootstrap  # noqa: F401
from dronomy_loc.config import load_config, resolve
from dronomy_loc.data.fetch import download_file, human_bytes


def _progress():
    state = {"last": -1}

    def cb(done, total):
        pct = int(done * 100 / total) if total else -1
        if pct != state["last"]:        # only redraw on a percent change
            state["last"] = pct
            bar = f"{pct:3d}%" if pct >= 0 else "  ? "
            tot = human_bytes(total) if total else "?"
            print(f"\r  {bar}  {human_bytes(done)} / {tot}", end="", flush=True)

    return cb


def main():
    cfg = load_config()
    url = getattr(cfg.video, "source_url", None)
    nbytes = getattr(cfg.video, "source_bytes", None)
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=url, help="source URL (default: config.video.source_url)")
    ap.add_argument("--out", default=cfg.video.path, help="destination path")
    ap.add_argument("--bytes", type=int, default=nbytes,
                    help="expected size in bytes (integrity check)")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    if not args.url:
        sys.exit("No source URL. Set video.source_url in config.yaml or pass --url.")

    dest = resolve(args.out)
    if args.force and dest.exists():
        dest.unlink()
    print(f"Fetching video -> {dest}")
    print(f"  source: {args.url.split('?')[0]}")
    download_file(args.url, dest, expected_bytes=args.bytes, progress=_progress())
    print()  # newline after the progress bar
    size = dest.stat().st_size
    ok = (args.bytes is None) or (size == args.bytes)
    print(f"Done: {dest}  ({human_bytes(size)}, {size} bytes)  "
          f"{'integrity OK' if ok else 'WARNING size mismatch'}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
