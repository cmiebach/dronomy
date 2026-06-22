"""Put `src/` on sys.path so the scripts can `import dronomy_loc` without an
install. (For a real install do `pip install -e .` using pyproject.toml.)"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
