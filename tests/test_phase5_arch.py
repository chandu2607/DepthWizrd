"""Phase-5 tests: the RECONSTRUCTION-FIDELITY architecture variant (SmallReconUNet).

Covers the §16 checklist for the ONE new variable -- the fusion-head architecture
(`arch`: "unet3" = original SmallFusionUNet, "unet4" = SmallReconUNet with one extra
encoder/decoder level):

  * output dimensions (B,H,W == input spatial dims; matches SmallFusionUNet)
  * shape consistency across several input sizes
  * skip-connection wiring (decoder conv in-channels == deep feature + encoder skip)
  * the NEW deepest level is present in unet4 and ABSENT in unet3 (e4 / d4)
  * one extra pooling stage (bottleneck 32->16 @ res 256): verified by a forward hook
  * parameter-count sanity: unet4 > unet3, still lightweight (< 1M), width UNCHANGED
    (the added params buy depth/receptive field, NOT raw width -- §9)
  * finite gradients (forward + masked-L1 backward: no NaN, no explosion)
  * transform compatibility (arch=unet4 + log1p trains + inverts, no NaN)
  * determinism (same seed -> bit-identical init on CPU)
  * REPRODUCIBILITY GUARD: arch="unet3" (and the default) stays byte-identical to the
    pre-Phase-5 SmallFusionUNet -> C_none / C_log1p remain valid controls (§10)
  * arch validation (unknown arch rejected; the ONLY difference vs C_log1p is `arch`)

The architecture is defined only under `if _HAS_TORCH:`, so every test skips cleanly
when torch is absent.

    python tests/test_phase5_arch.py      # or pytest
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.config import TrainConfig

try:
    import torch
    from depthwizard.models.fusion_head import (
        LearnedFusionHead, SmallFusionUNet, SmallReconUNet, _masked_l1)
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

W = 24  # the fixed base width (Phase-1..5); NEVER changed by this experiment (§5, §9)
# The measured Phase-1 SmallFusionUNet(w=24) parameter count. Hardcoded as a
# REPRODUCIBILITY GUARD: if the 3-level control's param count ever changes, C_log1p is
# no longer the same control and this test must fail loudly.
UNET3_PARAMS = 474_073


def _require_torch():
    if not _HAS_TORCH:
        print("  (torch absent -> architecture tests skipped)")
        return False
    return True


def _synthetic_sample(hw=48, sid="t0"):
    """A tiny scene with a moderate + a tall building (mirrors the Phase-4 fixture)."""
    rng = np.random.default_rng(0)
    rgb = (rng.random((hw, hw, 3)) * 255).astype(np.float32)
    depth = rng.random((hw, hw)).astype(np.float32)
    gt = np.zeros((hw, hw), dtype=np.float32)
    gt[8:16, 8:16] = 8.0     # moderate building
    gt[24:32, 24:32] = 40.0  # tall building (the reconstruction target)
    cls = np.full((hw, hw), 2, dtype=np.int32)
    cls[24:32, 24:32] = 6
    return {"rgb": rgb, "depth": depth, "gt": gt, "cls": cls, "city": "T", "id": sid}


# --------------------------------------------------------------- output dims / shape
def test_output_dims_match_input():
    """head(x): 4xHxW in -> BxHxW out; identical contract to SmallFusionUNet so the
    whole eval/resize path downstream is untouched (§16 output dimensions)."""
    if not _require_torch():
        return
    m = SmallReconUNet(w=W).eval()
    for b, hw in [(1, 256), (2, 128), (3, 64)]:
        x = torch.randn(b, 4, hw, hw)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (b, hw, hw), (y.shape, (b, hw, hw))


def test_shape_matches_between_architectures():
    """Same input -> same OUTPUT shape from unet3 and unet4 (only internal depth differs,
    never the I/O contract) -- §16 shape consistency."""
    if not _require_torch():
        return
    x = torch.randn(2, 4, 256, 256)
    with torch.no_grad():
        y3 = SmallFusionUNet(w=W).eval()(x)
        y4 = SmallReconUNet(w=W).eval()(x)
    assert y3.shape == y4.shape == (2, 256, 256)


def test_non_power_of_two_input_ok():
    """The decoder interpolates back to each encoder feature's size, so sizes that are
    not multiples of 16 still round-trip to the input resolution (robustness)."""
    if not _require_torch():
        return
    m = SmallReconUNet(w=W).eval()
    x = torch.randn(1, 4, 100, 130)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, 100, 130)


# --------------------------------------------------------------- new level / skips
def test_new_level_present_in_unet4_absent_in_unet3():
    """The single architectural change: unet4 gains one encoder (e4) and one decoder
    (d4) level; unet3 has neither (§16 new decoder level)."""
    if not _require_torch():
        return
    m4, m3 = SmallReconUNet(w=W), SmallFusionUNet(w=W)
    assert hasattr(m4, "e4") and hasattr(m4, "d4")
    assert not hasattr(m3, "e4") and not hasattr(m3, "d4")


def test_skip_connection_channel_wiring():
    """Each decoder block consumes deep-feature channels CONCATENATED with the encoder
    skip. Verify the in-channels arithmetic that encodes those skips (§16 skip conns).

    unet4:  d4 in = bottleneck(w*4) + e4(w*4) = w*8
            d3 in = d4(w*4)         + e3(w*4) = w*8
            d2 in = d3(w*2)         + e2(w*2) = w*4
            d1 in = d2(w)           + e1(w)   = w*2
    """
    if not _require_torch():
        return
    m = SmallReconUNet(w=W)
    assert m.d4[0].in_channels == W * 8 and m.d4[0].out_channels == W * 4
    assert m.d3[0].in_channels == W * 8 and m.d3[0].out_channels == W * 2
    assert m.d2[0].in_channels == W * 4 and m.d2[0].out_channels == W
    assert m.d1[0].in_channels == W * 2 and m.d1[0].out_channels == W
    # the new deepest encoder level is held at w*4 (NOT doubled to w*8): depth, not width.
    assert m.e4[0].in_channels == W * 4 and m.e4[0].out_channels == W * 4


def test_decoder_blocks_identical_to_unet3():
    """d3/d2/d1/head are SHARED with SmallFusionUNet unchanged -> unet4 adds a level, it
    does not redesign the decoder (keeps the change minimal, §7)."""
    if not _require_torch():
        return
    m4, m3 = SmallReconUNet(w=W), SmallFusionUNet(w=W)
    for name in ("d3", "d2", "d1"):
        a, b = getattr(m4, name), getattr(m3, name)
        assert a[0].in_channels == b[0].in_channels
        assert a[0].out_channels == b[0].out_channels
    assert m4.head.in_channels == m3.head.in_channels == W
    assert m4.head.out_channels == m3.head.out_channels == 1


def test_extra_pooling_stage_deepens_bottleneck():
    """The mechanism under test: one extra 2x pooling moves the bottleneck from 32x32
    (stride 8) to 16x16 (stride 16) at 256 px -> ~2x coarsest receptive field. Confirm
    it with a forward hook on the bottleneck block (§16 shape + the actual change)."""
    if not _require_torch():
        return
    cap = {}
    m4 = SmallReconUNet(w=W).eval()
    m3 = SmallFusionUNet(w=W).eval()
    h4 = m4.bottleneck.register_forward_hook(lambda _m, _i, o: cap.__setitem__("u4", o.shape[-2:]))
    h3 = m3.bottleneck.register_forward_hook(lambda _m, _i, o: cap.__setitem__("u3", o.shape[-2:]))
    with torch.no_grad():
        m4(torch.randn(1, 4, 256, 256))
        m3(torch.randn(1, 4, 256, 256))
    h4.remove(); h3.remove()
    assert tuple(cap["u3"]) == (32, 32), cap["u3"]   # stride 8
    assert tuple(cap["u4"]) == (16, 16), cap["u4"]   # stride 16 -> deeper context


# --------------------------------------------------------------- param-count sanity
def test_param_count_unet3_reproducibility_guard():
    """unet3 param count is FROZEN at the Phase-1 value. If this changes, C_log1p is no
    longer the same control (§10)."""
    if not _require_torch():
        return
    n3 = sum(p.numel() for p in SmallFusionUNet(w=W).parameters())
    assert n3 == UNET3_PARAMS, f"unet3 params drifted: {n3} != {UNET3_PARAMS}"


def test_param_count_unet4_larger_but_lightweight():
    """§9: unet4 must be BIGGER than unet3 (it added a level) yet still lightweight
    (< 1M params -- nowhere near a heavy backbone)."""
    if not _require_torch():
        return
    n3 = sum(p.numel() for p in SmallFusionUNet(w=W).parameters())
    n4 = sum(p.numel() for p in SmallReconUNet(w=W).parameters())
    assert n4 > n3, (n4, n3)
    assert n4 < 1_000_000, f"unet4 too heavy: {n4}"
    print(f"    [params] unet3={n3:,}  unet4={n4:,}  delta=+{n4 - n3:,} (+{100*(n4-n3)/n3:.0f}%)")


def test_width_unchanged_between_architectures():
    """The added capacity is DEPTH/receptive field, not width: the base width w=24 (the
    first encoder block's output channels) is identical in both nets (§5 no channel change)."""
    if not _require_torch():
        return
    m4, m3 = SmallReconUNet(w=W), SmallFusionUNet(w=W)
    assert m4.e1[0].out_channels == m3.e1[0].out_channels == W
    assert m4.e2[0].out_channels == m3.e2[0].out_channels == W * 2
    assert m4.e3[0].out_channels == m3.e3[0].out_channels == W * 4


# --------------------------------------------------------------- gradients
def test_finite_gradients_no_explosion():
    """Forward + masked-L1 backward on unet4: loss finite, every grad finite, global grad
    norm bounded (no explosion) -- §16 finite gradients / §15 no exploding grads."""
    if not _require_torch():
        return
    torch.manual_seed(0)
    m = SmallReconUNet(w=W).train()
    x = torch.randn(2, 4, 64, 64, requires_grad=False)
    target = torch.randn(2, 64, 64)
    mask = torch.ones(2, 64, 64, dtype=torch.bool)
    pred = m(x)
    loss = _masked_l1(pred, target, mask)
    assert torch.isfinite(loss)
    loss.backward()
    total = 0.0
    for p in m.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
        total += float(p.grad.norm()) ** 2
    assert total ** 0.5 < 1e4, f"grad norm exploded: {total ** 0.5}"


# --------------------------------------------------------------- head integration
def test_head_selects_architecture_by_config():
    """LearnedFusionHead picks the class from cfg.arch; default + 'unet3' -> SmallFusionUNet
    (controls unchanged), 'unet4' -> SmallReconUNet (§10 the ONE new variable)."""
    if not _require_torch():
        return
    default = LearnedFusionHead(TrainConfig(target_transform="log1p"), nodata=None, seed=0, device="cpu")
    u3 = LearnedFusionHead(TrainConfig(target_transform="log1p", arch="unet3"), nodata=None, seed=0, device="cpu")
    u4 = LearnedFusionHead(TrainConfig(target_transform="log1p", arch="unet4"), nodata=None, seed=0, device="cpu")
    assert isinstance(default.model, SmallFusionUNet)
    assert isinstance(u3.model, SmallFusionUNet)
    assert isinstance(u4.model, SmallReconUNet)
    assert u4.n_params() > u3.n_params()


def test_unknown_arch_rejected():
    """__init__ validation guards against typos / unsupported arch names."""
    if not _require_torch():
        return
    try:
        LearnedFusionHead(TrainConfig(arch="unet5"), nodata=None, seed=0, device="cpu")
    except ValueError:
        return
    raise AssertionError("unknown arch was not rejected")


# --------------------------------------------------------------- determinism / reproducibility
def test_unet4_deterministic_given_seed():
    """Same seed -> bit-identical unet4 init on CPU (so C_log1p_recon itself is
    reproducible across the two experiment seeds' construction) -- §16 determinism."""
    if not _require_torch():
        return
    torch.manual_seed(0); a = SmallReconUNet(w=W)
    torch.manual_seed(0); b = SmallReconUNet(w=W)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_unet3_path_bit_identical_reproducibility_preserved():
    """THE §10 guard: adding the arch selector must NOT perturb the unet3 path. Two
    seed-0 unet3 heads (default AND explicit) must be byte-identical to each other -- so
    C_none / C_log1p reproduce exactly as in Phase-1..4 (the arch getattr/validation
    touch no torch RNG before manual_seed+construction)."""
    if not _require_torch():
        return
    h_default = LearnedFusionHead(TrainConfig(target_transform="log1p"), nodata=None, seed=0, device="cpu")
    h_explicit = LearnedFusionHead(TrainConfig(target_transform="log1p", arch="unet3"), nodata=None, seed=0, device="cpu")
    ps_d = list(h_default.model.parameters())
    ps_e = list(h_explicit.model.parameters())
    assert len(ps_d) == len(ps_e)
    for a, b in zip(ps_d, ps_e):
        assert torch.equal(a, b), "unet3 init drifted -> C_log1p control no longer reproducible"


def test_recon_head_trains_and_inverts_log1p_without_nan():
    """Tiny end-to-end on the NEW arch: arch=unet4 + log1p fits a few synthetic tiles,
    params stay finite, prediction inverts to metric space and matches gt shape (§16
    transform compatibility; plumbing guard, NOT scientific evidence)."""
    if not _require_torch():
        return
    cfg = TrainConfig(target_transform="log1p", loss_type="standard", arch="unet4",
                      epochs=1, batch_size=2, train_res=32, amp=False)
    head = LearnedFusionHead(cfg, nodata=None, seed=0, device="cpu")
    assert isinstance(head.model, SmallReconUNet)
    samples = [_synthetic_sample(sid=f"t{i}") for i in range(3)]
    head.fit(samples)
    for p in head.model.parameters():
        assert torch.isfinite(p).all(), "non-finite parameter after unet4 training"
    pred = head.predict(samples[0])
    assert np.isfinite(pred).all() and pred.shape == samples[0]["gt"].shape


def test_arch_is_the_only_difference_vs_c_log1p():
    """Single-variable discipline (§5/§10): C_log1p (unet3) and C_log1p_recon (unet4) must
    share EVERY other training knob -- transform, loss, lr, epochs, batch, width, res."""
    if not _require_torch():
        return
    base = TrainConfig(target_transform="log1p", loss_type="standard", arch="unet3")
    recon = replace(base, arch="unet4")
    ctrl = LearnedFusionHead(base, nodata=None, seed=0, device="cpu")
    rec = LearnedFusionHead(recon, nodata=None, seed=0, device="cpu")
    for attr in ("target_transform", "loss_type", "loss_weight_scale", "loss_weight_max",
                 "loss_tail_start", "loss_tail_scale", "loss_tail_max"):
        assert getattr(ctrl, attr) == getattr(rec, attr), attr
    for attr in ("lr", "epochs", "batch_size", "width", "train_res"):
        assert getattr(ctrl.cfg, attr) == getattr(rec.cfg, attr), attr
    assert ctrl.arch == "unet3" and rec.arch == "unet4"  # the ONE intended difference


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nall {len(fns)} phase-5 architecture tests passed.")
