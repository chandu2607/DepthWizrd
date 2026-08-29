"""Phase-3 tests: the height-aware loss weighting for Baseline C.

Two layers:
  1. The weight function w(h) = min(1 + max(h,0)/scale, w_max) is a pure-numpy,
     transform-agnostic map on PHYSICAL height. It must be monotonic, bounded,
     ground-preserving (w(0)=1), stable on tall/huge/negative inputs.
  2. The weighted masked-L1 must reduce to the plain masked mean under uniform
     weights, respect the mask, up-weight tall pixels, and be built from the
     PHYSICAL (pre-log1p) target so no hidden log-space objective is introduced.

The loss/head tests require torch and skip cleanly if it is absent, so the
pure-numpy weight-function tests still run anywhere.

    python tests/test_phase3_loss.py      # or pytest
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.config import TrainConfig
from depthwizard.models.fusion_head import height_weight

try:
    import torch
    from depthwizard.models.fusion_head import (
        LearnedFusionHead, _masked_l1, _masked_weighted_l1)
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

SCALE, WMAX = 7.0, 5.0  # the Phase-3 defaults (scale ~ train building median)


def _require_torch():
    if not _HAS_TORCH:
        print("  (torch absent -> loss/head tests skipped)")
        return False
    return True


# --------------------------------------------------------- weight function (numpy)
def test_weight_zero_height_is_one():
    """Ground pixels keep weight exactly 1 -> rebalanced, never eliminated (§9)."""
    assert height_weight(0.0, SCALE, WMAX) == np.float32(1.0)
    assert np.allclose(height_weight(np.zeros(5), SCALE, WMAX), 1.0)


def test_weight_monotonic_nondecreasing():
    h = np.linspace(0, 200, 2001).astype(np.float32)
    w = height_weight(h, SCALE, WMAX)
    d = np.diff(w)
    assert (d >= -1e-7).all(), f"non-monotonic: min diff {d.min()}"


def test_weight_bounded():
    """w in [1, w_max] for the whole plausible range plus extremes."""
    h = np.array([-100, -1, 0, 0.5, 2, 5, 7, 10, 20, 40, 200, 1e6], dtype=np.float32)
    w = height_weight(h, SCALE, WMAX)
    assert w.min() >= 1.0 - 1e-6
    assert w.max() <= WMAX + 1e-6


def test_weight_saturates_at_cap():
    """Saturates at h = (w_max-1)*scale; equal weight for all taller pixels ->
    the 186 m train outlier cannot dominate (bounded, stable)."""
    h_sat = (WMAX - 1.0) * SCALE  # = 28 m for defaults
    assert np.isclose(height_weight(h_sat, SCALE, WMAX), WMAX, atol=1e-5)
    assert np.isclose(height_weight(1e5, SCALE, WMAX), WMAX, atol=1e-5)
    # just below saturation is strictly under the cap
    assert height_weight(h_sat - 1.0, SCALE, WMAX) < WMAX


def test_weight_negative_clamped():
    """Defensive: negative heights (should not occur post-resize) clamp to w=1."""
    assert np.isclose(height_weight(-5.0, SCALE, WMAX), 1.0)


def test_weight_numerical_stability_and_dtype():
    """No overflow/NaN on huge finite input; float32 out; shape preserved."""
    h = np.array([[0.0, 1e7], [3.4e38, 20.0]], dtype=np.float32)
    w = height_weight(h, SCALE, WMAX)
    assert np.isfinite(w).all()
    assert w.dtype == np.float32
    assert w.shape == h.shape


def test_weight_representative_heights():
    """Document the design points: median building (~7 m) counts 2x ground."""
    assert np.isclose(height_weight(7.0, SCALE, WMAX), 2.0, atol=1e-6)   # 1 + 7/7
    assert np.isclose(height_weight(2.0, SCALE, WMAX), 1 + 2/7, atol=1e-6)
    assert np.isclose(height_weight(5.0, SCALE, WMAX), 1 + 5/7, atol=1e-6)
    # tall bins approach / hit the cap
    assert height_weight(15.0, SCALE, WMAX) > height_weight(10.0, SCALE, WMAX)
    assert np.isclose(height_weight(40.0, SCALE, WMAX), WMAX)


def test_weight_scale_controls_slope():
    """A larger scale -> gentler up-weighting (configurable knob, §8)."""
    assert height_weight(10.0, 14.0, WMAX) < height_weight(10.0, 7.0, WMAX)


# --------------------------------------------------------- weighted masked loss (torch)
def test_masked_weighted_l1_uniform_equals_mean():
    if not _require_torch():
        return
    pred = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    target = torch.tensor([[1.5, 2.5, 1.0, 4.0]])
    mask = torch.tensor([[True, True, True, False]])
    w = torch.ones_like(pred)
    plain = _masked_l1(pred, target, mask)
    wtd = _masked_weighted_l1(pred, target, mask, w)
    assert torch.allclose(plain, wtd, atol=1e-6), (plain.item(), wtd.item())


def test_masked_weighted_l1_respects_mask():
    if not _require_torch():
        return
    # an invalid pixel with a huge error AND huge weight must be ignored.
    pred = torch.tensor([[0.0, 0.0]])
    target = torch.tensor([[1.0, 1000.0]])
    mask = torch.tensor([[True, False]])
    w = torch.tensor([[1.0, 50.0]])
    wtd = _masked_weighted_l1(pred, target, mask, w)
    assert torch.allclose(wtd, torch.tensor(1.0), atol=1e-6), wtd.item()


def test_masked_weighted_l1_upweights_tall():
    if not _require_torch():
        return
    # same |error| on two pixels; putting the weight on pixel B pulls the
    # weighted mean toward B's error relative to uniform (sanity of direction).
    pred = torch.tensor([[0.0, 0.0]])
    target = torch.tensor([[1.0, 3.0]])          # errors 1 and 3
    mask = torch.tensor([[True, True]])
    uni = _masked_weighted_l1(pred, target, mask, torch.ones_like(pred))
    tall = _masked_weighted_l1(pred, target, mask, torch.tensor([[1.0, 4.0]]))
    assert tall > uni, (uni.item(), tall.item())     # weighting the big error raises loss
    # weighted mean with w=[1,4]: (1*1 + 3*4)/(1+4) = 13/5 = 2.6
    assert torch.allclose(tall, torch.tensor(2.6), atol=1e-6), tall.item()


def test_masked_weighted_l1_empty_mask():
    if not _require_torch():
        return
    pred = torch.tensor([[1.0, 2.0]], requires_grad=True)
    target = torch.tensor([[0.0, 0.0]])
    mask = torch.tensor([[False, False]])
    out = _masked_weighted_l1(pred, target, mask, torch.ones_like(pred))
    assert torch.allclose(out, torch.tensor(0.0)), out.item()
    out.backward()  # must be differentiable / not crash on the empty guard


def _synthetic_sample(hw=48, sid="t0"):
    rng = np.random.default_rng(0)
    rgb = (rng.random((hw, hw, 3)) * 255).astype(np.float32)
    depth = rng.random((hw, hw)).astype(np.float32)
    gt = np.zeros((hw, hw), dtype=np.float32)
    gt[8:16, 8:16] = 8.0     # moderate building
    gt[24:32, 24:32] = 40.0  # tall building (above the ceiling)
    cls = np.full((hw, hw), 2, dtype=np.int32)
    cls[24:32, 24:32] = 6
    return {"rgb": rgb, "depth": depth, "gt": gt, "cls": cls, "city": "T", "id": sid}


def test_weight_built_from_physical_height_not_log_space():
    """THE key §6 test: with target_transform=log1p AND loss_type=height_weighted,
    the returned weight map must equal height_weight(PHYSICAL linear target), NOT
    height_weight(log1p target). Otherwise the loss would encode a hidden log-space
    importance and defeat the metric-space 'tall matters more' objective."""
    if not _require_torch():
        return
    cfg = TrainConfig(target_transform="log1p", loss_type="height_weighted",
                      loss_weight_scale=SCALE, loss_weight_max=WMAX)
    head = LearnedFusionHead(cfg, nodata=None, seed=0, device="cpu")
    # a standard-loss twin gives us the untransformed (linear) resized target.
    lin = LearnedFusionHead(replace(cfg, loss_type="standard", target_transform="none"),
                            nodata=None, seed=0, device="cpu")
    s = _synthetic_sample()
    t_log, m, w = head._prep_target(s, res=48)
    t_lin, _, w_lin_none = lin._prep_target(s, res=48)
    assert w is not None and w_lin_none is None
    # target is in log space ...
    assert np.allclose(t_log, np.log1p(np.maximum(t_lin, 0.0)), atol=1e-5)
    # ... but the WEIGHT is the physical-height weight (matches linear, not log).
    expected = height_weight(t_lin, SCALE, WMAX)
    assert np.allclose(w, expected, atol=1e-5)
    wrong = height_weight(t_log, SCALE, WMAX)   # what a log-space bug would give
    assert not np.allclose(w, wrong, atol=1e-2), "weight looks log-space (bug!)"


def test_height_weighted_head_trains_without_nan():
    """Tiny end-to-end: the weighted head fits a few synthetic tiles, loss finite,
    params stay finite (plumbing guard; NOT scientific evidence)."""
    if not _require_torch():
        return
    cfg = TrainConfig(target_transform="log1p", loss_type="height_weighted",
                      loss_weight_scale=SCALE, loss_weight_max=WMAX,
                      epochs=1, batch_size=2, train_res=32, amp=False)
    head = LearnedFusionHead(cfg, nodata=None, seed=0, device="cpu")
    samples = [_synthetic_sample(sid=f"t{i}") for i in range(3)]
    head.fit(samples)
    for p in head.model.parameters():
        assert torch.isfinite(p).all(), "non-finite parameter after weighted training"
    pred = head.predict(samples[0])
    assert np.isfinite(pred).all() and pred.shape == samples[0]["gt"].shape


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nall {len(fns)} phase-3 loss tests passed.")
