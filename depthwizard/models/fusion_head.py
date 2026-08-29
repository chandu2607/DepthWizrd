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

        def __init__(self, w: int = 24, in_channels: int = 4, out_channels: int = 1):
            super().__init__()
            self.e1 = _conv_block(in_channels, w)
            self.e2 = _conv_block(w, w * 2)
            self.e3 = _conv_block(w * 2, w * 4)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = _conv_block(w * 4, w * 4)
            self.d3 = _conv_block(w * 4 + w * 4, w * 2)
            self.d2 = _conv_block(w * 2 + w * 2, w)
            self.d1 = _conv_block(w + w, w)
            self.head = nn.Conv2d(w, out_channels, 1)

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
            out = self.head(d1)
            if out.shape[1] == 1:
                return out.squeeze(1)  # BxHxW
            return out # BxCxHxW

    class SmallReconUNet(nn.Module):
        """Phase-5 reconstruction-fidelity variant of SmallFusionUNet.

        The ONLY change vs SmallFusionUNet is DEPTH: one additional encoder/decoder
        level (a 4th 2x pooling stage), moving the bottleneck from 32x32 (stride 8)
        to 16x16 (stride 16) at train_res=256. This roughly DOUBLES the coarsest-scale
        effective receptive field (theoretical bottleneck RF ~66 -> ~138 px), so a
        single deep feature cell can integrate the FULL footprint of a large structure
        and the decoder can distribute a coherent height across its whole body instead
        of only lighting up the rim (the Phase-4 body-collapse / edge-overshoot failure).

        Design discipline (§7/§9): the new deepest level is held at w*4 channels -- it is
        NOT doubled to w*8 as a textbook U-Net would -- so the added parameters buy
        receptive field / multiscale reconstruction DEPTH, not raw width. Width w, input
        channels (4), output (1), and every decoder block shared with the 3-level net
        (d3/d2/d1/head) are unchanged, so this is a single-variable (add-one-level) test.
        """

        def __init__(self, w: int = 24, in_channels: int = 4, out_channels: int = 1):
            super().__init__()
            self.e1 = _conv_block(in_channels, w)
            self.e2 = _conv_block(w, w * 2)
            self.e3 = _conv_block(w * 2, w * 4)
            self.e4 = _conv_block(w * 4, w * 4)          # NEW deepest encoder level (held at 4w)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = _conv_block(w * 4, w * 4)  # now at 16x16 (was 32x32)
            self.d4 = _conv_block(w * 4 + w * 4, w * 4)  # NEW decoder level (skip = e4)
            self.d3 = _conv_block(w * 4 + w * 4, w * 2)  # identical to SmallFusionUNet.d3
            self.d2 = _conv_block(w * 2 + w * 2, w)      # identical
            self.d1 = _conv_block(w + w, w)              # identical
            self.head = nn.Conv2d(w, out_channels, 1)               # identical

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            e4 = self.e4(self.pool(e3))
            b = self.bottleneck(self.pool(e4))
            b = F.interpolate(b, size=e4.shape[-2:], mode="bilinear", align_corners=False)
            d4 = self.d4(torch.cat([b, e4], 1))
            d4 = F.interpolate(d4, size=e3.shape[-2:], mode="bilinear", align_corners=False)
            d3 = self.d3(torch.cat([d4, e3], 1))
            d3 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
            d2 = self.d2(torch.cat([d3, e2], 1))
            d2 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
            d1 = self.d1(torch.cat([d2, e1], 1))
            out = self.head(d1)
            if out.shape[1] == 1:
                return out.squeeze(1)  # BxHxW
            return out # BxCxHxW


def height_weight(h, scale: float, w_max: float):
    """Per-pixel loss weight as a function of PHYSICAL height h (meters).

        w(h) = min(1 + max(h, 0) / scale, w_max)

    Design (Phase-3, height-aware loss):
      * basis is PHYSICAL height, not the log1p-transformed target -- "tall matters
        more" is a metric-space objective, and a log-space basis would compress a
        40 m building to nearly the weight of a 15 m one (§6). Deriving it from the
        pre-transform target also makes it transform-agnostic (works for none/log1p).
      * ground (h=0) -> 1: ground is REBALANCED down in relative emphasis, never
        eliminated (§9); every valid pixel keeps weight >= 1.
      * monotonically increasing, smooth except at the cap knee.
      * bounded in [1, w_max]: the cap saturates at h = (w_max-1)*scale, so the rare
        tall outliers (train building max 186 m) cannot dominate / destabilize (§9,§22).
      * scale ~ training building-pixel median, w_max a small constant -> both derived
        ONLY from the training distribution (§4); reproducible and easy to explain.

    Pure numpy so it is testable without torch and is evaluated on the numpy target
    BEFORE tensor conversion. Works on scalars or arrays.
    """
    hh = np.maximum(np.asarray(h, dtype=np.float32), 0.0)
    w = 1.0 + hh / float(scale)
    return np.minimum(w, float(w_max)).astype(np.float32)


def tail_weight(h, h_start: float, tail_scale: float, w_max: float):
    """Per-pixel loss weight for a CALIBRATED TAIL objective (Phase-4).

        w(h) = 1                                          for h <= h_start
             = min(1 + (h - h_start)/tail_scale, w_max)   for h  > h_start

    Same design axioms as height_weight (PHYSICAL-height basis, bounded, monotone) but
    with a THRESHOLD: the weight is exactly 1 through the low/moderate regime and only
    rises past h_start. Motivation (Phase-3): the un-thresholded 1 + h/scale ramp
    up-weighted the abundant 0–15 m population (≈92% of JAX-train pixels) and shifted the
    whole prediction distribution up. Holding w=1 below h_start protects that regime while
    still emphasising the genuinely difficult tall tail.

      * h <= h_start (incl. h=0 and any negative slack) -> w=1 EXACTLY: the
        max(h-h_start, 0) clamp makes low / zero / negative heights numerically safe and
        leaves the dominant low/moderate regime's optimization emphasis unchanged.
      * monotonically non-decreasing; CONTINUOUS at h_start (w=1, no jump) -- a gentle
        piecewise-linear ramp (a slope kink, never a hard discontinuity) above it.
      * bounded in [1, w_max]: the cap saturates at h = h_start + (w_max-1)*tail_scale, so
        rare tall outliers (train building max ~186 m) cannot dominate the gradient (§11).
      * h_start / tail_scale / w_max are ALL derived from the JAX-train height distribution
        ONLY (see scripts/phase4_weight_diagnostic.py); no test-city leakage.

    Pure numpy (testable without torch), evaluated on the numpy target BEFORE the log1p
    transform (metric-space objective, transform-agnostic). Works on scalars or arrays.
    """
    over = np.maximum(np.asarray(h, dtype=np.float32) - float(h_start), 0.0)
    w = 1.0 + over / float(tail_scale)
    return np.minimum(w, float(w_max)).astype(np.float32)


def _masked_l1(pred, target, mask):
    """L1 over valid pixels only."""
    diff = torch.abs(pred - target)[mask]
    if diff.numel() == 0:
        return pred.sum() * 0.0
    return diff.mean()


def _masked_weighted_l1(pred, target, mask, weight):
    """Weighted L1 over valid pixels -- weighted MEAN (normalized by the sum of
    weights, NOT the pixel count).

    Normalizing by Sum(w) keeps the loss magnitude comparable to the standard
    masked-L1 (a convex combination of the same |residual| terms), so the effective
    gradient scale -- and thus the effective learning rate -- is UNCHANGED. The only
    thing that changes vs standard L1 is the RELATIVE emphasis across pixels. This is
    what makes the experiment a clean single-variable test (§7: same LR): a Sum(w)-count
    normalization would silently inflate the loss ~mean(w)x and act like an LR increase.

    `weight` is a per-pixel tensor (same shape as pred), >= 0.
    """
    diff = torch.abs(pred - target)
    w = weight[mask]
    if w.numel() == 0:
        return pred.sum() * 0.0
    num = (diff[mask] * w).sum()
    den = w.sum()
    return num / (den + 1e-8)


class LearnedFusionHead(HeightEstimator):
    name = "C_learned_fusion"
    metric = True

    def __init__(self, cfg_train, nodata: float | None = None, seed: int = 0, device: str | None = None):
        _require_torch()
        self.cfg = cfg_train
        self.nodata = nodata
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Phase-5: architecture selector (the ONE reconstruction variable under test).
        #   "unet3" (default) -> original 3-level SmallFusionUNet, so C_none / C_log1p
        #                        stay BIT-IDENTICAL to Phase-1..4.
        #   "unet4"           -> SmallReconUNet: one extra pooling level (bottleneck
        #                        stride 8->16), ~2x effective receptive field.
        # getattr + validation below touch NO torch RNG, so in the unet3 path model
        # construction still follows torch.manual_seed(seed) directly (reproducible).
        self.arch = getattr(cfg_train, "arch", "unet3")
        if self.arch not in ("unet3", "unet4"):
            raise ValueError(f"unknown arch: {self.arch!r}")
        torch.manual_seed(seed)
        # Phase-11: input mode ablation.
        self.input_mode = getattr(cfg_train, "input_mode", "rgb_depth")
        if self.input_mode not in ("rgb", "depth", "rgb_depth"):
            raise ValueError(f"unknown input_mode: {self.input_mode!r}")
            
        in_channels = 3 if self.input_mode == "rgb" else 1 if self.input_mode == "depth" else 4

        self.target_transform = getattr(cfg_train, "target_transform", "none")
        if self.target_transform not in ("none", "log1p", "classification"):
            raise ValueError(f"unknown target_transform: {self.target_transform!r}")
            
        out_channels = 8 if self.target_transform == "classification" else 1

        if self.arch == "unet4":
            self.model = SmallReconUNet(w=cfg_train.width, in_channels=in_channels, out_channels=out_channels).to(self.device)
        else:
            self.model = SmallFusionUNet(w=cfg_train.width, in_channels=in_channels, out_channels=out_channels).to(self.device)
        # Phase-3: optional height-aware loss weighting. "standard" reproduces the
        # plain masked-L1 exactly (default -> C_none / C_log1p unchanged).
        self.loss_type = getattr(cfg_train, "loss_type", "standard")
        if self.loss_type not in ("standard", "height_weighted", "tail_weighted"):
            raise ValueError(f"unknown loss_type: {self.loss_type!r}")
        self.loss_weight_scale = float(getattr(cfg_train, "loss_weight_scale", 7.0))
        self.loss_weight_max = float(getattr(cfg_train, "loss_weight_max", 5.0))
        # Phase-4 calibrated tail weight params (used only when loss_type=tail_weighted).
        self.loss_tail_start = float(getattr(cfg_train, "loss_tail_start", 15.0))
        self.loss_tail_scale = float(getattr(cfg_train, "loss_tail_scale", 12.5))
        self.loss_tail_max = float(getattr(cfg_train, "loss_tail_max", 3.0))
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
        
        if self.input_mode == "rgb":
            x = rgb.transpose(2, 0, 1)  # 3xHxW
        elif self.input_mode == "depth":
            x = depth[None]  # 1xHxW
        else:
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
        # Phase-3: derive the per-pixel loss weight from PHYSICAL height (gt_r here is
        # still in meters -- this is computed BEFORE the log1p transform below, so the
        # weight is transform-agnostic and encodes the metric-space "tall matters more"
        # objective, not a log-space one). None for the standard (unweighted) path.
        w_r = None
        if self.loss_type == "height_weighted":
            w_r = height_weight(gt_r, self.loss_weight_scale, self.loss_weight_max)
        elif self.loss_type == "tail_weighted":
            w_r = tail_weight(gt_r, self.loss_tail_start, self.loss_tail_scale,
                              self.loss_tail_max)
        if self.target_transform == "log1p":
            gt_r = np.log1p(np.maximum(gt_r, 0.0))
        elif self.target_transform == "classification":
            bins = np.array([0, 2, 5, 10, 15, 20, 30, 40, np.inf])
            gt_r = np.digitize(np.maximum(gt_r, 0.0), bins) - 1
            gt_r = np.clip(gt_r, 0, 7).astype(np.int64)
        return gt_r, valid_r, w_r

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
                xs, ys, ms, ws = [], [], [], []
                for j in idx:
                    xs.append(self._prep_xy(samples[j], res))
                    y, m, wj = self._prep_target(samples[j], res)
                    ys.append(y); ms.append(m)
                    if wj is not None:
                        ws.append(wj)
                x = torch.from_numpy(np.stack(xs)).float().to(self.device)
                if self.target_transform == "classification":
                    y = torch.from_numpy(np.stack(ys)).long().to(self.device)
                else:
                    y = torch.from_numpy(np.stack(ys)).float().to(self.device)
                m = torch.from_numpy(np.stack(ms)).bool().to(self.device)
                wt = None
                if self.loss_type in ("height_weighted", "tail_weighted"):
                    wt = torch.from_numpy(np.stack(ws)).float().to(self.device)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = self.model(x)
                    if self.target_transform == "classification":
                        # Standard Cross Entropy Loss over valid pixels
                        loss = F.cross_entropy(pred, y, reduction='none')
                        loss = loss[m].mean()
                    elif self.loss_type in ("height_weighted", "tail_weighted"):
                        loss = _masked_weighted_l1(pred, y, m, wt)
                    else:
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
        pred = self.model(xt).squeeze(0)
        if self.target_transform == "classification":
            pred = torch.argmax(pred, dim=0).cpu().numpy().astype(np.uint8)
        else:
            pred = pred.cpu().numpy().astype(np.float32)

        if self.target_transform == "log1p":
            # Invert to linear meters BEFORE resizing (so downstream metrics/resize
            # all operate in metric space, identical to the original C path).
            pred = np.expm1(pred)
        h, w = np.asarray(sample["gt"]).shape[:2]
        interp = cv2.INTER_NEAREST if self.target_transform == "classification" else cv2.INTER_LINEAR
        return cv2.resize(pred, (w, h), interpolation=interp)

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))
