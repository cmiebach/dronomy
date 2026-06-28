"""RoMA matcher running NATIVELY (no Docker) via the `romatch` package.

The Docker MatchAnything image runs RoMA on an emulated x86 CPU (~191 s/match).
This wrapper runs Edstedt's RoMa directly in the host Python, so on Apple Silicon
it uses the MPS GPU (~20 s/match — ~10x faster) and on an NVIDIA host it uses
CUDA. Same Matcher interface as the others, so it drops into the pipeline.

Note: this is the original `roma_outdoor` model (the SOTA dense matcher), not the
MatchAnything-finetuned checkpoint that ships only in the imcui Docker image.
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import Matcher, MatchResult, estimate_homography


class RomaMpsMatcher(Matcher):
    def __init__(self, cfg=None, device: str = "mps"):
        r = getattr(getattr(cfg, "matching", None), "ransac", None) if cfg else None
        self.device = device
        self.reproj = getattr(r, "reproj_threshold_px", 5.0)
        self.confidence = getattr(r, "confidence", 0.999)
        self.min_inliers = getattr(r, "min_inliers", 12)
        self._model = None

    def _model_(self):
        if self._model is None:
            from romatch import roma_outdoor
            # lower res keeps RoMA's dense matching inside 16 GB unified memory
            self._model = roma_outdoor(device=self.device,
                                       coarse_res=(448, 448), upsample_res=(560, 560))
        return self._model

    def match(self, drone_bgr: np.ndarray, ref_rgb: np.ndarray) -> MatchResult:
        from PIL import Image
        model = self._model_()
        a = Image.fromarray(cv2.cvtColor(drone_bgr, cv2.COLOR_BGR2RGB))
        b = ref_rgb if ref_rgb.ndim == 3 else cv2.cvtColor(ref_rgb, cv2.COLOR_GRAY2BGR)
        b = Image.fromarray(b if b.shape[2] == 3 else cv2.cvtColor(b, cv2.COLOR_BGR2RGB))
        Wa, Ha = a.size
        Wb, Hb = b.size
        warp, cert = model.match(a, b, device=self.device)
        matches, cert = model.sample(warp, cert)
        ka, kb = model.to_pixel_coordinates(matches, Ha, Wa, Hb, Wb)
        src = ka.detach().cpu().numpy().astype(np.float32)
        dst = kb.detach().cpu().numpy().astype(np.float32)
        if len(src) < 4:
            return MatchResult(src, dst, None, None, len(src))
        H, mask = estimate_homography(src, dst, self.reproj, self.confidence, self.min_inliers)
        try:                                   # free MPS memory between candidates (avoid OOM)
            import torch
            if self.device == "mps":
                torch.mps.empty_cache()
        except Exception:
            pass
        return MatchResult(src, dst, H, mask, len(src))
