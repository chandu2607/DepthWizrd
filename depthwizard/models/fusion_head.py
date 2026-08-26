"""Baseline C: frozen Depth Anything V2 + a SMALL learned RGB+depth fusion head.

This is the core hypothesis under test. Depth Anything V2 is used ONLY as a
frozen relative-depth prior (its output is precomputed and cached upstream);
this module learns a small CNN that maps [RGB(3) + relative-depth(1)] -> nDSM
(meters). It is intentionally tiny (a few hundred k params by default) so it
trains on student hardware / free Colab GPU in minutes, and so any measured
improvement over the affine baseline is attributable to *learned fusion*, not
model capacity.

The central Phase-1 question: does this learned head beat GlobalAffine (B),
*especially on a fully held-out city*? If not -> ABANDON (see eval/decision.py).

Requires torch. Import is guarded so numpy-only baselines (A/B) and the metrics
still work in environments without torch.
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np

from .base import HeightEstimator, Sample

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


def _require_torch():
    if not _HAS_TORCH:
        raise RuntimeError(
            "Baseline C (LearnedFusionHead) requires torch. "
            "Install torch or restrict baselines to ['A','B']."
        )


if _HAS_TORCH:

    def _conv_block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    class SmallFusionUNet(nn.Module):
        """Compact 3-level U-Net. in=4 (RGB+depth), out=1 (nDSM)."""

        def __init__(self, w: int = 24):
            super().__init__()
            self.e1 = _conv_block(4, w)
            self.e2 = _conv_block(w, w * 2)
            self.e3 = _conv_block(w * 2, w * 4)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = _conv_block(w * 4, w * 4)
            self.d3 = _conv_block(w * 4 + w * 4, w * 2)
            self.d2 = _conv_block(w * 2 + w * 2, w)
            self.d1 = _conv_block(w + w, w)
            self.head = nn.Conv2d(w, 1, 1)

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            b = self.bottleneck(self.pool(e3))
            b = F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False)
            d3 = self.d3(torch.cat([b, e3], 1))
            d3 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
            d2 = self.d2(torch.cat([d3, e2], 1))
            d2 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
            d1 = self.d1(torch.cat([d2, e1], 1))
            return self.head(d1).squeeze(1)  # BxHxW


def _masked_l1(pred, target, mask):
    """L1 over valid pixels only."""
    diff = torch.abs(pred - target)[mask]
    if diff.numel() == 0:
        return pred.sum() * 0.0
    return diff.mean()


class LearnedFusionHead(HeightEstimator):
    name = "C_learned_fusion"
    metric = True

    def __init__(self, cfg_train, nodata: float | None = None, seed: int = 0, device: str | None = None):
        _require_torch()
        self.cfg = cfg_train
        self.nodata = nodata
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        self.model = SmallFusionUNet(w=cfg_train.width).to(self.device)
        # Phase-2: optional target-space transform. "none" reproduces the original
        # Baseline C exactly; "log1p" trains in log-height space (inverted at predict).
        self.target_transform = getattr(cfg_train, "target_transform", "none")
        if self.target_transform not in ("none", "log1p"):
            raise ValueError(f"unknown target_transform: {self.target_transform!r}")
        # Depth normalization stats (filled during fit) so inputs are well-scaled.
        self.d_mean = 0.0
        self.d_std = 1.0

    # ---- data prep -------------------------------------------------------
    def _prep_xy(self, s: Sample, res: int):
        import cv2

        rgb = np.asarray(s["rgb"], dtype=np.float32)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        depth = np.asarray(s["depth"], dtype=np.float32)
        rgb = cv2.resize(rgb, (res, res), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (res, res), interpolation=cv2.INTER_LINEAR)
        depth = (depth - self.d_mean) / (self.d_std + 1e-6)
        x = np.concatenate([rgb.transpose(2, 0, 1), depth[None]], axis=0)  # 4xHxW
        return x

    def _prep_target(self, s: Sample, res: int):
        import cv2

        gt = np.asarray(s["gt"], dtype=np.float32)
        valid = np.isfinite(gt)
        if self.nodata is not None:
            valid &= gt != self.nodata
        gt_f = np.where(valid, gt, 0.0)
        gt_r = cv2.resize(gt_f, (res, res), interpolation=cv2.INTER_LINEAR)
        valid_r = cv2.resize(valid.astype(np.float32), (res, res),
                             interpolation=cv2.INTER_NEAREST) > 0.5
        if self.target_transform == "log1p":
            # Pointwise reparameterization of the SAME resized target the original
            # C uses -> the two variants differ ONLY by this transform (one variable).
            # nDSM is non-negative here; clamp defensively so log1p stays defined.
            gt_r = np.log1p(np.maximum(gt_r, 0.0))
        return gt_r, valid_r

    def fit(self, train_samples: Iterable[Sample]) -> "LearnedFusionHead":
        samples: List[Sample] = list(train_samples)
        # Global depth normalization from a subsample of training depth.
        ds = []
        for s in samples[: min(len(samples), 64)]:
            ds.append(np.asarray(s["depth"], dtype=np.float32).ravel()[::37])
        allc = np.concatenate(ds)
        self.d_mean, self.d_std = float(np.mean(allc)), float(np.std(allc) + 1e-6)

        res = self.cfg.train_res
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        use_amp = bool(self.cfg.amp and self.device == "cuda")
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        rng = np.random.default_rng(self.seed)
        bs = self.cfg.batch_size

        self.model.train()
        for epoch in range(self.cfg.epochs):
            order = rng.permutation(len(samples))
            ep_loss, nb = 0.0, 0
            for i in range(0, len(order), bs):
                idx = order[i : i + bs]
                if idx.size < 2:
                    # BatchNorm needs >1 sample per channel in train mode; a lone
                    # trailing tile would crash. Dropping it costs nothing here.
                    continue
                xs, ys, ms = [], [], []
                for j in idx:
                    xs.append(self._prep_xy(samples[j], res))
                    y, m = self._prep_target(samples[j], res)
                    ys.append(y); ms.append(m)
                x = torch.from_numpy(np.stack(xs)).float().to(self.device)
                y = torch.from_numpy(np.stack(ys)).float().to(self.device)
                m = torch.from_numpy(np.stack(ms)).bool().to(self.device)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = self.model(x)
                    loss = _masked_l1(pred, y, m)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                ep_loss += float(loss.detach()); nb += 1
            print(f"[C][seed={self.seed}] epoch {epoch+1}/{self.cfg.epochs} "
                  f"masked-L1={ep_loss/max(nb,1):.3f}")
        return self

    @torch.no_grad()
    def predict(self, sample: Sample) -> np.ndarray:
        import cv2

        self.model.eval()
        res = self.cfg.train_res
        x = self._prep_xy(sample, res)
        xt = torch.from_numpy(x[None]).float().to(self.device)
        pred = self.model(xt).squeeze(0).cpu().numpy().astype(np.float32)
        if self.target_transform == "log1p":
            # Invert to linear meters BEFORE resizing (so downstream metrics/resize
            # all operate in metric space, identical to the original C path).
            pred = np.expm1(pred)
        h, w = np.asarray(sample["gt"]).shape[:2]
        return cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))
