"""Fake relative-depth provider for OFFLINE smoke tests ONLY.

When torch / transformers / a GPU are unavailable (e.g. a laptop just checking
the plumbing), the real Depth Anything V2 wrapper cannot run. This provider
fabricates a depth map from the reference nDSM (a blurred, noised, z-scored copy)
so the A/B/C code paths execute end-to-end. It is NOT a model and its outputs are
NOT feasibility evidence -- any run using it is marked evidence_valid=False and
the report says so in bold. Use only with --allow-fake-depth.
"""
from __future__ import annotations

import numpy as np


class FakeDepth:
    is_fake = True

    def __init__(self, noise: float = 0.35, seed: int = 0):
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def infer_from_gt(self, gt: np.ndarray, target_hw) -> np.ndarray:
        import cv2
        g = np.asarray(gt, dtype=np.float32)
        g = np.where(np.isfinite(g), g, 0.0)
        g = cv2.GaussianBlur(g, (0, 0), sigmaX=2.0)
        mu, sd = float(g.mean()), float(g.std() + 1e-6)
        z = (g - mu) / sd
        z = z + self.rng.normal(0, self.noise, z.shape).astype(np.float32)
        return cv2.resize(z, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
