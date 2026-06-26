"""Job B figure: model-comparison plot renders to a PNG (headless Agg)."""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronomy_loc.viz.figures import fig_model_comparison  # noqa: E402


def test_model_comparison_writes_png(tmp_path):
    per = {
        "sift": types.SimpleNamespace(recall_5m=0.20, lock_rate=0.50, median_err_m=30.0),
        "loftr": types.SimpleNamespace(recall_5m=0.80, lock_rate=0.90, median_err_m=2.0),
        "roma": types.SimpleNamespace(recall_5m=0.95, lock_rate=1.00, median_err_m=1.5),
    }
    out = fig_model_comparison(per, tmp_path / "cmp.png", title="Model comparison")
    assert out.exists() and out.stat().st_size > 1000


def test_model_comparison_handles_missing_median(tmp_path):
    # A model that never locked has median_err_m=None — must not crash.
    per = {"sift": types.SimpleNamespace(recall_5m=0.0, lock_rate=0.0, median_err_m=None)}
    out = fig_model_comparison(per, tmp_path / "cmp2.png")
    assert out.exists()
