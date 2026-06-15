"""MatchAnything matcher (zju3dv) — same `Matcher` interface as SIFT/LoFTR.

MatchAnything is trained for *cross-modality* matching (drone vs. satellite,
across season/sensor/lighting), which is exactly this project's bottleneck: the
limiter here is not accuracy on matchable frames (LoFTR already hits ~1.7 m) but
COVERAGE — only ~6% of frames match the orthophoto at all because of the
appearance gap. MatchAnything is the lever aimed at that gap; success = the
matchable fraction rises above 6% on the 35-stop scan, not a lower error on the
already-matchable frames.

Real MatchAnything weights (`matchanything_eloftr` / `matchanything_roma`) ship
only in zju3dv's fork of `imcui` (image-matching-webui); the PyPI `imcui` pins
numpy<2.3 and falls back to the base architecture. So this backend is meant to
run in the dedicated weights environment (see docker/Dockerfile.matchanything),
NOT the main env — the import is deferred so the rest of the package, and the
test suite, never depend on it. Drop-in once that env is active:
`get_matcher("matchanything", cfg)`.
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import Matcher, MatchResult, estimate_homography


def _build_api(model: str, device: str):
    """Construct imcui's ImageMatchingAPI for a MatchAnything model. Isolated
    in one function so the heavy import is lazy and tests can monkeypatch it."""
    try:
        import os
        from importlib.resources import files

        from imcui.api import ImageMatchingAPI
        from imcui.ui.utils import get_matcher_zoo, load_config
    except ImportError as exc:  # pragma: no cover - exercised only without imcui
        raise ImportError(
            "MatchAnything needs 'imcui' with zju3dv's MatchAnything weights — "
            "use the dedicated env (docker/Dockerfile.matchanything). "
            f"(import error: {exc})"
        ) from exc

    cfg_path = os.environ.get("IMCUI_CONFIG")
    if not cfg_path:
        for rel in ("config/config.yaml", "config/app.yaml"):
            p = files("imcui").joinpath(rel)
            if p.is_file():
                cfg_path = str(p)
                break
    zoo = get_matcher_zoo(load_config(cfg_path)["matcher_zoo"])
    lower = {k.lower(): k for k in zoo}
    name = next((lower[c] for c in (f"matchanything_{model}".lower(), model.lower())
                 if c in lower), None)
    if name is None:
        raise RuntimeError(f"No imcui zoo entry for {model!r}; have {sorted(zoo)}")
    return ImageMatchingAPI(conf=zoo[name], device=device), name


def _extract_mkpts(pred: dict) -> tuple[np.ndarray, np.ndarray]:
    """Pull corresponding points out of imcui's prediction dict (key name
    varies by version); we run our own RANSAC, so the interface stays uniform."""
    for k0, k1 in (("mkeypoints0_orig", "mkeypoints1_orig"),
                   ("mkpts0", "mkpts1"), ("mkpts0_f", "mkpts1_f")):
        if k0 in pred and k1 in pred:
            return (np.asarray(pred[k0], np.float32), np.asarray(pred[k1], np.float32))
    raise KeyError(f"no match points in imcui output; keys={sorted(pred)}")


class MatchAnythingMatcher(Matcher):
    def __init__(self, cfg=None, model: str | None = None):
        m = getattr(getattr(cfg, "matching", None), "matchanything", None) if cfg else None
        r = getattr(getattr(cfg, "matching", None), "ransac", None) if cfg else None
        self.model = model or getattr(m, "model", "eloftr")     # eloftr | roma
        self.device = getattr(m, "device", "cpu")
        self.max_long_edge = getattr(m, "max_long_edge", 832)
        self.reproj = getattr(r, "reproj_threshold_px", 5.0)
        self.confidence = getattr(r, "confidence", 0.999)
        self.min_inliers = getattr(r, "min_inliers", 12)
        self._api = None
        self._zoo_name = None

    def _prep(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        """BGR/any ndarray -> uint8 RGB (H,W,3) for imcui; longest edge capped.
        imcui silently returns zero matches on float[0,1] input. Returns the
        scale so keypoints can be mapped back to the caller's pixel space."""
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        scale = min(1.0, self.max_long_edge / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (round(w * scale), round(h * scale)),
                             interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), scale

    def match(self, drone_bgr: np.ndarray, ref_rgb: np.ndarray) -> MatchResult:
        if self._api is None:
            self._api, self._zoo_name = _build_api(self.model, self.device)
        rgb0, s0 = self._prep(drone_bgr)
        rgb1, s1 = self._prep(ref_rgb)
        pred = self._api(rgb0, rgb1)
        src, dst = _extract_mkpts(pred)
        src, dst = src / s0, dst / s1            # back to caller pixel coords
        if len(src) < 4:
            return MatchResult(src, dst, None, None, len(src))
        H, mask = estimate_homography(src, dst, self.reproj, self.confidence,
                                      self.min_inliers)
        return MatchResult(src, dst, H, mask, len(src))
