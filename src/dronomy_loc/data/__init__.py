"""Data loading: drone video frame extraction + camera intrinsics.

NOTE: `intrinsics` is intentionally NOT re-exported here — it imports
`framework.schema`, which imports `data.telemetry`, so re-exporting it from this
package `__init__` creates a circular import. Import it directly:
`from dronomy_loc.data.intrinsics import intrinsics_from_config`.
"""
from .frames import extract_frames, iter_frames, FrameInfo  # noqa: F401
