"""Phase-4 tests: the CALIBRATED TAIL-WEIGHTED loss for Baseline C.

Covers the §16 checklist for tail_weight(h, h_start, tail_scale, w_max):
  * low height -> weight ~1 (flat through the protected 0–15 m regime)
  * monotonicity (non-decreasing)
  * threshold behaviour (w=1 at/below h_start; >1 just above)
  * smoothness (CONTINUOUS at h_start -- no jump/discontinuity)
  * boundedness ([1, w_max])
  * numerical stability (huge/finite inputs, dtype, shape)
  * negative / zero-height safety (w=1)
  * tall-height emphasis (rises above threshold, saturates at the cap)
  * transformed-target compatibility + correct PHYSICAL-height basis (§15)
  * it is strictly GENTLER than the Phase-3 aggressive height_weight

The loss/head tests require torch and skip cleanly if it is absent, so the
pure-numpy weight-function tests still run anywhere.

    python tests/test_phase4_loss.py      # or pytest
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.config import TrainConfig
from depthwizard.models.fusion_head import height_weight, tail_weight

try:
    import torch
    from depthwizard.models.fusion_head import (
        LearnedFusionHead, _masked_l1, _masked_weighted_l1)
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

# Phase-4 calibrated-tail defaults, ALL derived from the JAX-train height distribution:
#   h_start=15 -> onset of the sparse tail (measured ~P92 all / ~P79 building; ≈ the ~14 m ceiling)
#   scale=12.5 -> ramp reaches the cap at 15 + (3-1)*12.5 = 40 m ≈ P99.3
#   w_max=3    -> gentler than the aggressive height_weight cap (5.0)
H_START, T_SCALE, T_MAX = 15.0, 12.5, 3.0
H_SAT = H_START + (T_MAX - 1.0) * T_SCALE  # = 40 m: height at which w hits the cap
# The Phase-3 aggressive weight, for the "strictly gentler" comparison.
AGG_SCALE, AGG_MAX = 7.0, 5.0


def _require_torch():
    if not _HAS_TORCH:
        print("  (torch absent -> loss/head tests skipped)")
        return False
    return True


# --------------------------------------------------------- weight function (numpy)
def test_low_height_weight_is_exactly_one():
    """The whole point of calibration: the protected 0–15 m regime stays at w=1
    EXACTLY (unlike height_weight, which up-weights it and shifted the distribution)."""
    for h in [0.0, 0.5, 2.0, 5.0, 7.16, 10.0, 14.0, 15.0]:
        assert tail_weight(h, H_START, T_SCALE, T_MAX) == np.float32(1.0), h
    assert np.allclose(tail_weight(np.linspace(0, 15, 64), H_START, T_SCALE, T_MAX), 1.0)


def test_threshold_behaviour():
    """w=1 for h<=h_start, and strictly >1 immediately above it."""
    assert tail_weight(H_START, H_START, T_SCALE, T_MAX) == np.float32(1.0)
    assert tail_weight(H_START - 1e-3, H_START, T_SCALE, T_MAX) == np.float32(1.0)
    assert tail_weight(H_START + 0.5, H_START, T_SCALE, T_MAX) > 1.0
    assert tail_weight(H_START + 5.0, H_START, T_SCALE, T_MAX) > 1.0


def test_continuous_at_threshold_no_jump():
    """'Smooth ramp preferred over a hard discontinuity' (§9): w is continuous at
    h_start -- approaching from above tends to w=1, no jump."""
    just_above = tail_weight(H_START + 1e-4, H_START, T_SCALE, T_MAX)
    assert abs(float(just_above) - 1.0) < 1e-3, just_above
    # and the ramp is gradual, not a step: a small step in h -> a small step in w.
    hs = np.linspace(H_START, H_SAT, 500).astype(np.float32)
    dw = np.diff(tail_weight(hs, H_START, T_SCALE, T_MAX))
    step = (hs[1] - hs[0]) / T_SCALE
    assert np.all(dw >= -1e-7) and np.all(dw <= step + 1e-6)


def test_monotonic_nondecreasing():
    h = np.linspace(0, 200, 2001).astype(np.float32)
    w = tail_weight(h, H_START, T_SCALE, T_MAX)
    d = np.diff(w)
    assert (d >= -1e-7).all(), f"non-monotonic: min diff {d.min()}"


def test_bounded():
    """w in [1, w_max] across the whole plausible range plus extremes."""
    h = np.array([-100, -1, 0, 5, 15, 20, 40, 100, 200, 1e6], dtype=np.float32)
    w = tail_weight(h, H_START, T_SCALE, T_MAX)
    assert w.min() >= 1.0 - 1e-6
    assert w.max() <= T_MAX + 1e-6


def test_saturates_at_cap():
    """Saturates at h_start+(w_max-1)*scale (=40 m); equal weight for all taller
    pixels so the ~186 m train outlier cannot dominate the gradient (§11)."""
    assert np.isclose(tail_weight(H_SAT, H_START, T_SCALE, T_MAX), T_MAX, atol=1e-5)
    assert np.isclose(tail_weight(1e5, H_START, T_SCALE, T_MAX), T_MAX, atol=1e-5)
    assert tail_weight(H_SAT - 1.0, H_START, T_SCALE, T_MAX) < T_MAX


def test_negative_and_zero_safety():
    """Negative heights (should not occur post-resize) and zero clamp to w=1."""
    assert np.isclose(tail_weight(-5.0, H_START, T_SCALE, T_MAX), 1.0)
    assert np.isclose(tail_weight(0.0, H_START, T_SCALE, T_MAX), 1.0)
    assert np.allclose(tail_weight(np.array([-9, -1, 0], np.float32),
                                   H_START, T_SCALE, T_MAX), 1.0)


def test_numerical_stability_and_dtype():
    """No overflow/NaN on huge finite input; float32 out; shape preserved."""
    h = np.array([[0.0, 1e7], [3.4e38, 20.0]], dtype=np.float32)
    w = tail_weight(h, H_START, T_SCALE, T_MAX)
    assert np.isfinite(w).all()
    assert w.dtype == np.float32
    assert w.shape == h.shape


def test_tall_emphasis_representative_heights():
    """Document the design points: mild for moderate-tall, meaningful (not extreme)
    for very tall, capped for the rare extremes."""
    assert np.isclose(tail_weight(20.0, H_START, T_SCALE, T_MAX), 1 + 5 / 12.5)   # 1.40
    assert np.isclose(tail_weight(25.0, H_START, T_SCALE, T_MAX), 1 + 10 / 12.5)  # 1.80
    assert np.isclose(tail_weight(30.0, H_START, T_SCALE, T_MAX), 1 + 15 / 12.5)  # 2.20
    assert np.isclose(tail_weight(40.0, H_START, T_SCALE, T_MAX), T_MAX)          # 3.00
    # strictly increasing through the tall regime
    assert (tail_weight(30.0, H_START, T_SCALE, T_MAX)
            > tail_weight(20.0, H_START, T_SCALE, T_MAX)
            > tail_weight(16.0, H_START, T_SCALE, T_MAX))


def test_strictly_gentler_than_aggressive_weight():
    """The calibration hypothesis, made testable: the tail weight never exceeds the
    Phase-3 aggressive weight, and is STRICTLY smaller across the 0–15 m regime it is
    designed to protect (there height_weight>1 but tail_weight==1)."""
    h = np.linspace(0, 200, 4001).astype(np.float32)
    t = tail_weight(h, H_START, T_SCALE, T_MAX)
    a = height_weight(h, AGG_SCALE, AGG_MAX)
    assert np.all(t <= a + 1e-6), "tail weight exceeds the aggressive weight somewhere"
    low = (h > 0) & (h <= H_START)
    assert np.all(t[low] == 1.0)
    assert np.all(a[low] > 1.0)          # aggressive up-weights the protected regime
    assert np.all(t[low] < a[low])       # tail is strictly gentler there


def test_scale_and_start_are_knobs():
    """Bigger tail_scale -> gentler ramp; bigger h_start -> later onset (both §8 knobs)."""
    assert (tail_weight(25.0, H_START, 25.0, T_MAX)
            < tail_weight(25.0, H_START, 12.5, T_MAX))     # gentler slope
    assert (tail_weight(18.0, 20.0, T_SCALE, T_MAX)
            < tail_weight(18.0, 15.0, T_SCALE, T_MAX))     # later onset -> still 1 at 18


# --------------------------------------------------------- weighted masked loss (torch)
def test_tail_weighted_reuses_weighted_l1_uniform_equals_mean():
    """Sanity that the tail weight rides the SAME weighted-mean masked-L1 (uniform
    weights reduce it to the plain masked mean) -- so effective LR is unchanged."""
    if not _require_torch():
        return
    pred = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    target = torch.tensor([[1.5, 2.5, 1.0, 4.0]])
    mask = torch.tensor([[True, True, True, False]])
    plain = _masked_l1(pred, target, mask)
    wtd = _masked_weighted_l1(pred, target, mask, torch.ones_like(pred))
    assert torch.allclose(plain, wtd, atol=1e-6), (plain.item(), wtd.item())


def _synthetic_sample(hw=48, sid="t0"):
    rng = np.random.default_rng(0)
    rgb = (rng.random((hw, hw, 3)) * 255).astype(np.float32)
    depth = rng.random((hw, hw)).astype(np.float32)
    gt = np.zeros((hw, hw), dtype=np.float32)
    gt[8:16, 8:16] = 8.0     # moderate building -> should get w=1 (protected regime)
    gt[24:32, 24:32] = 40.0  # tall building (above threshold) -> up-weighted
    cls = np.full((hw, hw), 2, dtype=np.int32)
    cls[24:32, 24:32] = 6
    return {"rgb": rgb, "depth": depth, "gt": gt, "cls": cls, "city": "T", "id": sid}


def test_weight_built_from_physical_height_not_log_space():
    """THE key §15 test: with target_transform=log1p AND loss_type=tail_weighted, the
    returned weight map must equal tail_weight(PHYSICAL linear target), NOT
    tail_weight(log1p target). A log-space basis would compress the tall tail and
    defeat the metric-space 'tall matters more' objective."""
    if not _require_torch():
        return
    cfg = TrainConfig(target_transform="log1p", loss_type="tail_weighted",
                      loss_tail_start=H_START, loss_tail_scale=T_SCALE, loss_tail_max=T_MAX)
    head = LearnedFusionHead(cfg, nodata=None, seed=0, device="cpu")
    lin = LearnedFusionHead(replace(cfg, loss_type="standard", target_transform="none"),
                            nodata=None, seed=0, device="cpu")
    s = _synthetic_sample()
    t_log, m, w = head._prep_target(s, res=48)
    t_lin, _, w_none = lin._prep_target(s, res=48)
    assert w is not None and w_none is None
    # target is in log space ...
    assert np.allclose(t_log, np.log1p(np.maximum(t_lin, 0.0)), atol=1e-5)
    # ... but the WEIGHT is the physical-height tail weight (matches linear, not log).
    expected = tail_weight(t_lin, H_START, T_SCALE, T_MAX)
    assert np.allclose(w, expected, atol=1e-5)
    wrong = tail_weight(t_log, H_START, T_SCALE, T_MAX)  # what a log-space bug would give
    assert not np.allclose(w, wrong, atol=1e-2), "weight looks log-space (bug!)"
    # concretely: the 8 m 'moderate building' block stays at w=1 (protected), the 40 m
    # tall block is up-weighted -- the calibration in action, on the real target path.
    assert np.isclose(float(w[t_lin == 8.0].mean()), 1.0, atol=1e-5)
    assert float(w[t_lin >= 39.0].mean()) > 1.0


def test_standard_path_unweighted_reproducibility_preserved():
    """Adding tail_weighted must NOT perturb the standard path: loss_type=standard
    still returns weight=None -> C_none / C_log1p remain byte-identical controls."""
    if not _require_torch():
        return
    for tf in ("none", "log1p"):
        cfg = TrainConfig(target_transform=tf, loss_type="standard")
        head = LearnedFusionHead(cfg, nodata=None, seed=0, device="cpu")
        _, _, w = head._prep_target(_synthetic_sample(), res=48)
        assert w is None, f"standard path produced weights for transform={tf}"


def test_unknown_loss_type_rejected():
    """__init__ validation still guards against typos / unsupported modes."""
    if not _require_torch():
        return
    try:
        LearnedFusionHead(TrainConfig(loss_type="tail_wieghted"),  # deliberate typo
                          nodata=None, seed=0, device="cpu")
    except ValueError:
        return
    raise AssertionError("unknown loss_type was not rejected")


def test_tail_weighted_head_trains_without_nan():
    """Tiny end-to-end: the tail-weighted head fits a few synthetic tiles, loss finite,
    params stay finite, prediction inverts (plumbing guard; NOT scientific evidence)."""
    if not _require_torch():
        return
    cfg = TrainConfig(target_transform="log1p", loss_type="tail_weighted",
                      loss_tail_start=H_START, loss_tail_scale=T_SCALE, loss_tail_max=T_MAX,
                      epochs=1, batch_size=2, train_res=32, amp=False)
    head = LearnedFusionHead(cfg, nodata=None, seed=0, device="cpu")
    samples = [_synthetic_sample(sid=f"t{i}") for i in range(3)]
    head.fit(samples)
    for p in head.model.parameters():
        assert torch.isfinite(p).all(), "non-finite parameter after tail-weighted training"
    pred = head.predict(samples[0])
    assert np.isfinite(pred).all() and pred.shape == samples[0]["gt"].shape


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nall {len(fns)} phase-4 tail-loss tests passed.")
