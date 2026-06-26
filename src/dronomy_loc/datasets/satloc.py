"""SatLoc dataset adapter — STRETCH GOAL stub.

SatLoc (https://github.com/ameth64/SatLoc) is a GNSS-denied UAV localization
dataset: per-sortie drone image sequences plus georeferenced satellite DOM
`.tif` tiles, with `ref_sample.csv` mapping tile sequence numbers to geographic
coordinates and an `eval_samplel.py` helper for tile->coordinate conversion.

Implementation sketch (when prioritised): mirror `uavvisloc.py` — yield one
`Scenario` per sortie; build the `reference()` `FetchTile` from the DOM tiles +
`ref_sample.csv` extents (lon/lat corners -> EPSG:3857 -> `GeoImage` ->
`make_world_fetch`); read drone frames into `Sample`s with GT from the
filename-encoded lon/lat. Until then this raises clearly so the registry path
stays valid without pretending to support the dataset.
"""
from __future__ import annotations

from .base import Dataset
from ..framework.schema import Scenario


class SatLocDataset(Dataset):
    def __init__(self, cfg=None):
        self.cfg = cfg

    def scenarios(self) -> list[Scenario]:
        raise NotImplementedError(
            "SatLocDataset is a stretch-goal stub. Implement per the module "
            "docstring (mirror datasets/uavvisloc.py), or use 'video'/'uavvisloc'."
        )
