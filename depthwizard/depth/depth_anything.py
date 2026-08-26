"""Frozen Depth Anything V2 wrapper (the relative-depth prior).

Depth Anything V2 is used strictly as a FROZEN structural/relative-depth prior.
It is NOT a metric height sensor: its output is dimensionless, scale- and
shift-ambiguous, and for near-nadir overhead imagery the true camera-depth
variation across a scene is negligible (~0.1-0.2 per-mille of the sensor
distance), so the model is expressing learned appearance priors, not measured
geometry. We cache its output to disk so the (relatively expensive) inference
runs once per tile and every baseline reuses it.

Convention: for the relative model, larger output ~= nearer to camera ~=
higher for overhead views, so we expect a POSITIVE correlation with nDSM.
The affine / learned heads are free to invert the sign if needed.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np


class DepthAnythingV2:
    def __init__(self, model_id: str, input_size: int = 518,
                 cache_dir: str | None = None, use_cache: bool = True):
        self.model_id = model_id
        self.input_size = input_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache and cache_dir is not None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._pipe = None  # lazy so importing this module never loads torch/HF

    def _ensure_pipe(self):
        if self._pipe is None:
            import torch
            from transformers import pipeline
            device = 0 if torch.cuda.is_available() else -1
            self._pipe = pipeline("depth-estimation", model=self.model_id, device=device)
            self._device = "cuda" if device == 0 else "cpu"
            print(f"[depth] loaded {self.model_id} on {self._device}")

    def _cache_path(self, key: str) -> Path:
        h = hashlib.md5(f"{self.model_id}|{self.input_size}|{key}".encode()).hexdigest()
        return self.cache_dir / f"{h}.npy"

    def infer(self, rgb: np.ndarray, key: str, target_hw: tuple[int, int]) -> np.ndarray:
        """Return relative depth resized to target_hw=(H,W). Cached by `key`."""
        import cv2

        if self.use_cache:
            cp = self._cache_path(key)
            if cp.exists():
                d = np.load(cp)
                if d.shape == tuple(target_hw):
                    return d
                return cv2.resize(d, (target_hw[1], target_hw[0]),
                                  interpolation=cv2.INTER_LINEAR)

        from PIL import Image
        self._ensure_pipe()
        arr = np.asarray(rgb)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8) if arr.max() > 1.5 \
                else (arr * 255).astype(np.uint8)
        pil = Image.fromarray(arr[..., :3])
        if self.input_size and max(pil.size) != self.input_size:
            pil = pil.resize((self.input_size, self.input_size))
        out = self._pipe(pil)
        pd = out["predicted_depth"]
        try:
            import torch
            if isinstance(pd, torch.Tensor):
                pd = pd.detach().cpu().float().numpy()
        except Exception:
            pd = np.asarray(pd, dtype=np.float32)
        pd = np.squeeze(pd).astype(np.float32)
        d = cv2.resize(pd, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)

        if self.use_cache:
            np.save(self._cache_path(key), d)
        return d
