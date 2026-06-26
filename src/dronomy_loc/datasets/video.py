"""VideoDataset: the provided drone flight as ONE standardized `Scenario`.

This is the adapter that lets the generic engine run on the original DJI clip
without knowing anything video-specific. It maps the existing ingest reuse APIs
(`frames.iter_frames_sharpest`, `telemetry.load_track_csv`) onto the schema:
each sampled frame becomes a `Sample`, the GPS track (if present) populates
`Sample.gt` for SCORING ONLY, and the reference accessor is the same `FetchTile`
the localizer already consumes.

Two laziness rules keep construction cheap and side-effect-free:
  * `sample_iter` is a zero-arg FACTORY, not an iterator — every call replays the
    flight from frame zero, so a scenario can be scored more than once.
  * `fetch_tile` builds NO network at construction. We prefer the cached world
    tile (cropped locally, zero per-call network) and fall back to the live
    provider, but `TileCache`/`get_provider` only touch the network when the
    returned `FetchTile` is actually called.
"""
from __future__ import annotations

from pathlib import Path

from ..config import resolve
from ..data import frames
from ..data.telemetry import GPSFix, gt_for_frame, load_track_csv
from ..framework.schema import CameraIntrinsics, Sample, Scenario
from ..localize.search import TileCache
from ..localize.validate import make_world_fetch
from ..reference.base import get_provider
from ..reference.store import load_reference
from .base import Dataset


class VideoDataset(Dataset):
    """One raw drone video -> exactly one `Scenario`."""

    def __init__(self, cfg):
        self.cfg = cfg

    def scenarios(self) -> list[Scenario]:
        cfg = self.cfg
        video_path = resolve(cfg.video.path)
        name = video_path.stem                         # scenario name == filename stem
        terrain = getattr(cfg.video, "terrain", "mixed")
        prior = (cfg.video.rough_lat, cfg.video.rough_lon)
        intr = CameraIntrinsics(
            focal_px=getattr(cfg.camera, "focal_px", 3713.0),
            hfov_deg=getattr(cfg.camera, "hfov_deg", 84.0),
        )

        # GROUND TRUTH ONLY — loaded ONCE, never fed to localization. A missing
        # track must never crash the adapter: gt simply stays None per sample.
        track_csv = resolve(cfg.video.gps_track_csv)
        track: list[GPSFix] | None = (
            load_track_csv(track_csv) if Path(track_csv).exists() else None)

        every_n_seconds = cfg.frames.every_n_seconds
        resize_long_edge = getattr(cfg.frames, "resize_long_edge", 1920)

        def sample_iter():
            # Fresh stream each call (re-iterable scenario). Lazy: frames are
            # decoded on demand, so nothing is read until the runner iterates.
            for fi in frames.iter_frames_sharpest(
                video_path,
                every_n_seconds=every_n_seconds,
                resize_long_edge=resize_long_edge,
            ):
                yield Sample(
                    frame_id=fi.index,
                    image_bgr=fi.image,
                    t_s=fi.t_seconds,
                    gt=gt_for_frame(track, fi.index) if track else None,
                    intrinsics=intr,
                    meta={"blur_score": fi.blur_score, "source": name},
                )

        fetch_tile = self._build_fetch_tile()

        return [Scenario(
            name=name,
            terrain=terrain,
            fetch_tile=fetch_tile,
            sample_iter=sample_iter,
            prior=prior,
            intrinsics=intr,
            meta={"dataset": "video", "source": name, "prior": prior},
        )]

    def _build_fetch_tile(self):
        """Prefer the cached one-world tile (cropped locally, zero per-call
        network); fall back to the live provider. Neither path hits the network
        here — the wrapped `FetchTile` does, only when called."""
        cfg = self.cfg
        try:
            world = load_reference(
                resolve(cfg.reference.out_dir), f"world_{cfg.reference.provider}")
            return TileCache(make_world_fetch(world))
        except FileNotFoundError:
            return TileCache(get_provider(cfg.reference.provider, cfg).fetch)
