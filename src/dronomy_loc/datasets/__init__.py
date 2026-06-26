"""Dataset adapters: each turns a raw data source into standardized `Scenario`s
(the framework's plug-and-play data layer). Selected by name via `get_dataset`."""
from .base import Dataset, get_dataset  # noqa: F401
