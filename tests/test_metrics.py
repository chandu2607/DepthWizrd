"""Unit tests for the height metrics (numpy-only, run with pytest or directly).

    pytest tests/test_metrics.py       # or:  python tests/test_metrics.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.metrics.height_metrics import (
    valid_mask, compute_metrics, compute_class_metrics, aggregate_scene_metrics,
    compute_binned_metrics, aggregate_binned,
)


def test_perfect_prediction():
    gt = np.array([[0.0, 1.0], [2.0, 3.0]])
    m = compute_metrics(gt.copy(), gt)
    assert m.mae == 0.0
    assert m.rmse == 0.0
    assert abs(m.pearson - 1.0) < 1e-9
    assert m.n_pixels == 4


def test_constant_offset():
    gt = np.array([[0.0, 1.0], [2.0, 3.0]])
    pred = gt + 2.0
    m = compute_metrics(pred, gt)
    assert abs(m.mae - 2.0) < 1e-9
    assert abs(m.rmse - 2.0) < 1e-9
    assert abs(m.pearson - 1.0) < 1e-9  # perfectly correlated despite bias


def test_nodata_and_nan_masking():
    gt = np.array([[5.0, -999.0], [np.nan, 3.0]])
    pred = np.array([[5.0, 100.0], [100.0, 3.0]])
    m = compute_metrics(pred, gt, nodata=-999.0)
    assert m.n_pixels == 2          # sentinel and NaN dropped
    assert m.mae == 0.0             # remaining pixels match exactly


def test_zero_variance_pearson_is_nan():
    gt = np.array([[1.0, 2.0], [3.0, 4.0]])
    pred = np.full_like(gt, 7.0)    # constant prediction -> undefined correlation
    m = compute_metrics(pred, gt)
    assert np.isnan(m.pearson)
    assert m.n_pixels == 4


def test_empty_returns_nan_not_raise():
    gt = np.full((2, 2), np.nan)
    m = compute_metrics(np.zeros((2, 2)), gt)
    assert m.n_pixels == 0
    assert np.isnan(m.mae) and np.isnan(m.rmse)


def test_class_split_building():
    gt = np.array([[10.0, 10.0], [0.0, 0.0]])
    pred = np.array([[8.0, 12.0], [0.0, 0.0]])
    cls = np.array([[6, 6], [2, 2]])  # top row buildings, bottom row ground
    out = compute_class_metrics(pred, gt, cls, building_label=6)
    assert out["building"]["n_pixels"] == 2
    assert abs(out["building"]["mae"] - 2.0) < 1e-9
    assert out["non_building"]["n_pixels"] == 2
    assert out["non_building"]["mae"] == 0.0


def test_shape_mismatch_raises():
    try:
        compute_metrics(np.zeros((2, 2)), np.zeros((3, 3)))
    except ValueError:
        return
    raise AssertionError("expected ValueError on shape mismatch")


def test_aggregate_pooled():
    s1 = compute_metrics(np.array([2.0, 2.0]), np.array([0.0, 0.0])).as_dict()  # mae 2
    s2 = compute_metrics(np.array([4.0]), np.array([0.0])).as_dict()            # mae 4
    agg = aggregate_scene_metrics([s1, s2])
    assert agg["n_scenes"] == 2
    # pixel-weighted pooled MAE = (2*2 + 4*1) / 3 = 8/3
    assert abs(agg["mae_pooled"] - (8.0 / 3.0)) < 1e-9


def test_valid_mask_extra():
    gt = np.array([[1.0, 2.0], [3.0, 4.0]])
    extra = np.array([[True, False], [True, True]])
    m = valid_mask(gt, extra_mask=extra)
    assert m.sum() == 3


def test_binned_assigns_and_bias():
    # edges -> bins [0,2),[2,5),[5,inf). One pixel per bin, +1 m over-prediction.
    gt = np.array([1.0, 3.0, 10.0])
    pred = gt + 1.0
    b = compute_binned_metrics(pred, gt, edges=[0.0, 2.0, 5.0])
    assert [x["n_pixels"] for x in b] == [1, 1, 1]
    assert (b[0]["lo"], b[0]["hi"]) == (0.0, 2.0)
    assert b[-1]["hi"] == float("inf")
    for x in b:
        assert abs(x["mae"] - 1.0) < 1e-9
        assert abs(x["bias"] - 1.0) < 1e-9      # over-prediction -> positive bias


def test_binned_ceiling_shows_negative_bias():
    # a 14 m ceiling: tall GT, capped prediction -> strong NEGATIVE bias up high.
    gt = np.array([2.0, 30.0])
    pred = np.array([2.0, 14.0])                 # tall pixel saturates at ceiling
    b = compute_binned_metrics(pred, gt, edges=[0.0, 5.0, 20.0])
    top = b[-1]                                  # [20, inf) bin
    assert top["n_pixels"] == 1
    assert abs(top["bias"] - (-16.0)) < 1e-9     # 14 - 30
    assert abs(top["mae"] - 16.0) < 1e-9


def test_binned_empty_bin_is_nan():
    b = compute_binned_metrics(np.array([1.0]), np.array([1.0]), edges=[0.0, 2.0, 5.0])
    assert b[0]["n_pixels"] == 1
    assert b[1]["n_pixels"] == 0 and np.isnan(b[1]["mae"])


def test_aggregate_binned_pools_by_count():
    # two scenes into the same single bin [0,inf): mae 2 over 2 px, mae 4 over 1 px
    s1 = compute_binned_metrics(np.array([2.0, 2.0]), np.array([0.0, 0.0]), edges=[0.0])
    s2 = compute_binned_metrics(np.array([4.0]), np.array([0.0]), edges=[0.0])
    agg = aggregate_binned([s1, s2])
    assert len(agg) == 1
    assert agg[0]["n_pixels"] == 3
    # pooled MAE = (2*2 + 4*1)/3 = 8/3  (NOT the (2+4)/2 average of per-scene means)
    assert abs(agg[0]["mae"] - (8.0 / 3.0)) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nall {len(fns)} tests passed.")
