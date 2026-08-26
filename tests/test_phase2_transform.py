"""Phase-2 tests: the log1p target transform for Baseline C.

Two things must hold for the modification to be trustworthy:
  1. The transform is a clean, invertible reparameterization: the log-space
     target is exactly log1p of the SAME resized linear target the original C
     uses, and expm1 recovers it. (numpy-level, no training.)
  2. predict() inverts it: a model emitting a log-space value v yields expm1(v)
     metric meters, while the untransformed head passes the value through.

The head requires torch; those tests skip cleanly if torch is absent so the
numpy-only round-trip still runs anywhere.

    python tests/test_phase2_transform.py      # or pytest
"""
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.config import TrainConfig

try:
    import torch
    import torch.nn as nn
    from depthwizard.models.fusion_head import LearnedFusionHead
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


def test_log1p_expm1_roundtrip_pure():
    """The mathematical identity the transform relies on, over the real range."""
    h = np.array([0.0, 0.26, 7.16, 14.0, 45.0, 186.5], dtype=np.float64)  # obs. range
    assert np.allclose(np.expm1(np.log1p(h)), h, atol=1e-6)


def _synthetic_sample(hw=64, sid="t0"):
    rng = np.random.default_rng(0)
    rgb = (rng.random((hw, hw, 3)) * 255).astype(np.float32)
    depth = rng.random((hw, hw)).astype(np.float32)
    # a target with real dynamic range incl. tall structure above the ceiling
    gt = np.zeros((hw, hw), dtype=np.float32)
    gt[10:20, 10:20] = 8.0
    gt[30:40, 30:40] = 45.0
    cls = np.full((hw, hw), 2, dtype=np.int32)
    cls[30:40, 30:40] = 6  # building
    return {"rgb": rgb, "depth": depth, "gt": gt, "cls": cls, "city": "T", "id": sid}


def _require_torch():
    if not _HAS_TORCH:
        print("  (torch absent -> head tests skipped)")
        return False
    return True


def test_prep_target_log1p_matches_log_of_linear():
    if not _require_torch():
        return
    cfg_none = TrainConfig(target_transform="none")
    cfg_log = replace(cfg_none, target_transform="log1p")
    h_none = LearnedFusionHead(cfg_none, nodata=None, seed=0, device="cpu")
    h_log = LearnedFusionHead(cfg_log, nodata=None, seed=0, device="cpu")
    s = _synthetic_sample()
    t_none, m_none = h_none._prep_target(s, res=64)
    t_log, m_log = h_log._prep_target(s, res=64)
    # log target is exactly log1p of the identical resized linear target ...
    assert np.allclose(t_log, np.log1p(np.maximum(t_none, 0.0)), atol=1e-5)
    # ... expm1 recovers it, and the valid mask is untouched by the transform.
    assert np.allclose(np.expm1(t_log), t_none, atol=1e-4)
    assert np.array_equal(m_none, m_log)


def test_predict_inverts_log_space():
    if not _require_torch():
        return

    class _ConstModel(nn.Module):
        """Emit a constant value of shape B×H×W (stands in for a trained net)."""
        def __init__(self, c):
            super().__init__()
            self.c = float(c)

        def forward(self, x):
            b, _, h, w = x.shape
            return torch.full((b, h, w), self.c)

    s = _synthetic_sample()
    target_m = 20.0

    # log1p head: model emits log1p(20); predict must expm1 back to ~20 m.
    h_log = LearnedFusionHead(replace(TrainConfig(), target_transform="log1p"),
                              nodata=None, seed=0, device="cpu")
    h_log.model = _ConstModel(math.log1p(target_m))
    pred_log = h_log.predict(s)
    assert pred_log.shape == s["gt"].shape
    assert np.allclose(pred_log, target_m, atol=1e-3), pred_log.mean()

    # none head: model emits 20 directly; predict passes it through unchanged.
    h_none = LearnedFusionHead(TrainConfig(target_transform="none"),
                               nodata=None, seed=0, device="cpu")
    h_none.model = _ConstModel(target_m)
    pred_none = h_none.predict(s)
    assert np.allclose(pred_none, target_m, atol=1e-3)


def test_invalid_transform_rejected():
    if not _require_torch():
        return
    try:
        LearnedFusionHead(TrainConfig(target_transform="sqrt"),
                          nodata=None, seed=0, device="cpu")
    except ValueError:
        return
    raise AssertionError("expected ValueError on unknown target_transform")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nall {len(fns)} phase-2 transform tests passed.")
