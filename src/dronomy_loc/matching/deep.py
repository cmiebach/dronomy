"""Deep matcher: LoFTR (detector-free) via kornia; SuperGlue optional — STUB-ready.

LoFTR is well-suited to our hard case (low-texture grass, cross-domain drone↔
satellite) because it is detector-free and produces dense semi-correspondences.
kornia ships pretrained LoFTR weights ('outdoor') that auto-download on first use.

torch/kornia are listed in requirements.txt but may not be installed yet — the
import is deferred so the rest of the package works without them.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from .base import Matcher, MatchResult, estimate_homography


def _ensure_ca_bundle() -> None:
    """kornia downloads the LoFTR weights via torch.hub (urllib over TLS). On
    some systems — notably a fresh Windows Python — ssl can't locate a CA bundle
    and the download dies with CERTIFICATE_VERIFY_FAILED. Point it at certifi's
    bundle (a transitive dep of requests, always present) unless the user has
    already set one. ssl reads SSL_CERT_FILE when the request builds its context,
    so setting it here, before the first download, is enough."""
    if os.environ.get("SSL_CERT_FILE"):
        return
    try:
        import certifi
    except ImportError:  # pragma: no cover
        return
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())


class DeepMatcher(Matcher):
    def __init__(self, cfg=None, model: str = "loftr"):
        d = getattr(getattr(cfg, "matching", None), "deep", None) if cfg else None
        r = getattr(getattr(cfg, "matching", None), "ransac", None) if cfg else None
        self.model_name = getattr(d, "model", model)
        self.weights = getattr(d, "weights", "outdoor")
        self.device = getattr(d, "device", "cpu")
        self.reproj = getattr(r, "reproj_threshold_px", 5.0)
        self.confidence = getattr(r, "confidence", 0.999)
        self.min_inliers = getattr(r, "min_inliers", 12)
        self.max_long_edge = 1024  # LoFTR is heavy; cap input size (esp. on CPU)
        self._model = None
        self._torch = None

    def _lazy_init(self):
        if self._model is not None:
            return
        try:
            import torch
            import kornia as K
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "DeepMatcher needs torch + kornia. Install with:\n"
                "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n"
                "  pip install kornia"
            ) from e
        self._torch = torch
        _ensure_ca_bundle()  # so torch.hub can fetch the LoFTR weights over TLS
        self._model = K.feature.LoFTR(pretrained=self.weights).to(self.device).eval()

    def _prep(self, img: np.ndarray):
        """BGR/RGB ndarray -> 1x1xHxW float tensor in [0,1], grayscale, capped size."""
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        scale = min(1.0, self.max_long_edge / max(h, w))
        if scale < 1.0:
            gray = cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        t = self._torch.from_numpy(gray)[None, None].float() / 255.0
        return t.to(self.device), scale

    def match(self, drone_bgr: np.ndarray, ref_rgb: np.ndarray) -> MatchResult:
        self._lazy_init()
        t1, s1 = self._prep(drone_bgr)
        t2, s2 = self._prep(ref_rgb)
        with self._torch.inference_mode():
            out = self._model({"image0": t1, "image1": t2})
        src = out["keypoints0"].cpu().numpy() / s1   # back to original frame scale
        dst = out["keypoints1"].cpu().numpy() / s2
        if len(src) < 4:
            return MatchResult(src, dst, None, None, len(src))
        H, mask = estimate_homography(src, dst, self.reproj, self.confidence, self.min_inliers)
        return MatchResult(src, dst, H, mask, len(src))
