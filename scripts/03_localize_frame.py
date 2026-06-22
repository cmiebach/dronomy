"""Small working piece #3 (the MVP): localize a single drone frame against the
reference tile, print the estimated lat/lon/yaw, and save overlays.

Usage:
    python scripts/03_localize_frame.py --frame data/frames/frame_000600_t0020020ms.jpg
    python scripts/03_localize_frame.py --frame <img> --method loftr
"""
import argparse

import _bootstrap  # noqa: F401
import cv2

from dronomy_loc.config import load_config, resolve
from dronomy_loc.matching import get_matcher
from dronomy_loc.localize import localize_frame
from dronomy_loc.reference import load_reference
from dronomy_loc.viz import draw_matches, draw_frame_footprint


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, help="path to a drone frame image")
    ap.add_argument("--method", default=cfg.matching.method,
                    choices=["classical", "loftr", "superglue"])
    ap.add_argument("--provider", default=cfg.reference.provider)
    args = ap.parse_args()

    ref = load_reference(resolve(cfg.reference.out_dir), args.provider)
    frame = cv2.imread(args.frame)
    if frame is None:
        raise FileNotFoundError(args.frame)

    matcher = get_matcher(args.method, cfg)
    pose, mr = localize_frame(frame, ref, matcher)

    print(f"matcher={args.method}  matches={mr.n_matches}  inliers={mr.n_inliers}")
    if pose.ok:
        print(f"  lat={pose.lat:.6f}  lon={pose.lon:.6f}  "
              f"yaw={pose.yaw_deg:.1f} deg  ground={pose.ground_m_per_px:.3f} m/px")
    else:
        print("  FAILED to estimate a homography (too few inliers).")
        return

    out_dir = resolve(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "match_overlay.jpg"), draw_matches(frame, ref.image, mr))
    cv2.imwrite(str(out_dir / "footprint.jpg"),
                draw_frame_footprint(ref.image, mr.homography, frame.shape))
    print(f"Saved overlays to {out_dir}")


if __name__ == "__main__":
    main()
