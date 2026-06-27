"""Localization models: a uniform `localize()` over the matcher + grid search,
selected by name via `get_model`."""
from .base import (  # noqa: F401
    LocalizationModel, SceneSearch, get_model, search_for_altitude, MODEL_NAMES,
)
