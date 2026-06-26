"""Dataset adapter interface + factory (Inversion of Control for the data side).

A `Dataset` knows how to read one raw source (the provided video, UAV-VisLoc,
SatLoc, ...) and yields one or more standardized `Scenario`s — a single-flight
source yields one scenario; a multi-region benchmark yields one per region.
`get_dataset(name, cfg)` is the registry the config-driven runner selects from,
mirroring the existing `get_matcher` / `get_provider` factories.
"""
from __future__ import annotations

import abc

from ..framework.schema import Scenario


class Dataset(abc.ABC):
    """Reads a raw source and standardizes it into `Scenario`s."""

    @abc.abstractmethod
    def scenarios(self) -> list[Scenario]:
        """Return every localizable scenario this source provides."""
        raise NotImplementedError


def get_dataset(name: str, cfg=None) -> Dataset:
    """Factory: 'video' | 'uavvisloc' | 'satloc'. `cfg` is the loaded config
    namespace (optional)."""
    name = name.lower().replace("-", "").replace("_", "")
    if name in ("video", "flight"):
        from .video import VideoDataset
        return VideoDataset(cfg)
    if name in ("uavvisloc", "visloc"):
        from .uavvisloc import UAVVisLocDataset
        return UAVVisLocDataset(cfg)
    if name == "satloc":
        from .satloc import SatLocDataset
        return SatLocDataset(cfg)
    raise ValueError(
        f"Unknown dataset: {name!r} (expected 'video', 'uavvisloc' or 'satloc')")
