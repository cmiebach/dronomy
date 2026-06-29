"""ONE command, fresh machine -> full result set. The end-to-end entrypoint.

Chains the whole project in dependency order so a brand-new clone produces the
localization outputs with a single command:

  0. fetch     download the drone video                 (scripts/00_fetch_video)
  1. ingest    sharded, resumable, verified frames       (data/ingest.py)
  2. gps       decode the DJI GPS ground-truth track      (needs exiftool)
  3. reference fetch the per-provider world tiles         (reference/*)
  4. localize  run every matcher, auto-select, export     (scripts/run_all.py)

Each stage is idempotent and independently skippable (`--skip-*`), so a re-run
resumes instead of redoing finished work. exiftool is auto-provisioned on
Windows (portable, no admin); GPS is ground-truth scoring only, never an input.

Usage:
    python scripts/run_e2e.py                         # full run, sane defaults
    python scripts/run_e2e.py --blur off --spread 20  # fast ingest, 20 frames
    python scripts/run_e2e.py --skip-fetch --skip-ingest   # data already local
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from dronomy_loc.config import load_config, resolve
from dronomy_loc.data import ingest
from dronomy_loc.data.fetch import download_file, human_bytes
from dronomy_loc.data.telemetry import extract_gps_track
from dronomy_loc.reference import get_provider, load_reference, save_reference

HERE = Path(__file__).resolve().parent


def _hr(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}", flush=True)


def step_fetch(cfg, args):
    _hr("STEP 0/4: fetch video (data ingestion)")
    url = args.url or getattr(cfg.video, "source_url", None)
    dest = resolve(cfg.video.path)
    if not url:
        if dest.exists():
            print(f"no source_url set, but video already present: {dest}")
            return
        sys.exit("No video and no video.source_url to fetch it from.")
    nbytes = getattr(cfg.video, "source_bytes", None)
    download_file(url, dest, expected_bytes=nbytes)
    print(f"video ready: {dest} ({human_bytes(dest.stat().st_size)})")


def step_ingest(cfg, args):
    _hr("STEP 1/4: ingest and shard")
    video = resolve(cfg.video.path)
    out = resolve(getattr(cfg.frames, "ingest_dir", "data/ingest"))
    res = ingest.ingest_video(
        video, out,
        every_n_seconds=args.every,
        shard_seconds=getattr(cfg.frames, "shard_seconds", 30.0),
        blur_filter=args.blur,
        resize_long_edge=getattr(cfg.frames, "resize_long_edge", 1920),
        jpeg_quality=getattr(cfg.frames, "jpeg_quality", 95),
        max_frames=args.max_frames,
    )
    state = "complete" if res.completed else "PARTIAL (re-run to resume)"
    print(f"ingest {state}: {res.n_shards} shards | {res.n_frames_written} written "
          f"| {res.n_frames_skipped} skipped\nmanifest: {res.manifest_path}")


def step_gps(cfg, args):
    _hr("STEP 2/4: extract GPS ground truth track")
    from _ensure_tools import ensure_exiftool
    exe = ensure_exiftool(resolve("tools"))
    video = resolve(cfg.video.path)
    out = resolve(getattr(cfg.video, "gps_track_csv", "data/gps_track.csv"))
    fixes = extract_gps_track(video, out_csv=out, exiftool=exe)
    if not fixes:
        sys.exit("no GPS fixes decoded, cannot score without ground truth")
    print(f"extracted {len(fixes)} GPS fixes -> {out}")
    print(f"  first frame {fixes[0].frame} ({fixes[0].lat:.6f}, {fixes[0].lon:.6f}) "
          f"| last frame {fixes[-1].frame} ({fixes[-1].lat:.6f}, {fixes[-1].lon:.6f})")


def step_reference(cfg, args):
    _hr("STEP 3/4: fetch reference world tiles")
    ref_dir = resolve(cfg.reference.out_dir)
    span = getattr(cfg.reference, "world_span_m", 600.0)
    pix = getattr(cfg.reference, "world_pixels", 4096)
    ok = []
    for prov in [p.strip() for p in args.providers.split(",") if p.strip()]:
        name = f"world_{prov}"
        try:
            load_reference(ref_dir, name)
            print(f"  {prov}: cached ({name})")
            ok.append(prov)
            continue
        except FileNotFoundError:
            pass
        try:
            print(f"  {prov}: fetching {span:g} m / {pix}px world tile ...", flush=True)
            world = get_provider(prov, cfg).fetch(
                cfg.video.rough_lat, cfg.video.rough_lon, span, pix)
            save_reference(world, ref_dir, name)
            ok.append(prov)
        except Exception as e:
            print(f"  {prov}: UNAVAILABLE ({type(e).__name__}: {str(e)[:120]})")
    if not ok:
        sys.exit("no reference provider succeeded, cannot localize")
    args._providers_ok = ",".join(ok)
    print(f"providers ready: {args._providers_ok}")


def step_localize(cfg, args):
    _hr("STEP 4/4: localize (run_all: every matcher, auto select, export)")
    providers = getattr(args, "_providers_ok", None) or args.providers
    cmd = [sys.executable, str(HERE / "run_all.py"),
           "--providers", providers, "--methods", args.methods,
           "--device", args.device, "--spread", str(args.spread)]
    print("->", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        sys.exit(f"run_all.py failed (rc={rc})")


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None, help="override video source URL")
    ap.add_argument("--every", type=float, default=getattr(cfg.frames, "every_n_seconds", 1.0))
    ap.add_argument("--blur", choices=["sharpest", "off"],
                    default=getattr(cfg.frames, "blur_filter", "sharpest"))
    ap.add_argument("--max-frames", type=int, default=None, help="cap frames written this run")
    ap.add_argument("--providers", default="pnoa,esri")
    ap.add_argument("--methods", default="sift,loftr,roma")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--spread", type=int, default=20)
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--skip-ingest", action="store_true")
    ap.add_argument("--skip-gps", action="store_true")
    ap.add_argument("--skip-reference", action="store_true")
    ap.add_argument("--skip-localize", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    if not args.skip_fetch:
        step_fetch(cfg, args)
    if not args.skip_ingest:
        step_ingest(cfg, args)
    if not args.skip_gps:
        step_gps(cfg, args)
    if not args.skip_reference:
        step_reference(cfg, args)
    if not args.skip_localize:
        step_localize(cfg, args)
    _hr(f"END TO END COMPLETE in {time.perf_counter()-t0:.0f}s, "
        f"see data/outputs/run_all/RESULTS.md")


if __name__ == "__main__":
    main()
