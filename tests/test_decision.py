"""Tests for the GO/MODIFY/ABANDON decision logic.

Feeds decide() constructed metric dicts (shaped exactly like a real run's
results) to confirm every verdict branch fires correctly. No torch/data needed.

    python tests/test_decision.py     # or pytest
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.eval.decision import decide


def _agg(mae=None, rmse=None, r=None, n=10):
    a = {"n_scenes": n}
    if mae is not None:
        a["mae_pooled"] = mae
    if rmse is not None:
        a["rmse_pooled"] = rmse
    if r is not None:
        a["pearson_mean"] = r
    return a


def _mk(a_r_id, b_x_mae, c_x_mae, c_x_rmse, c_id_mae, b_id_mae,
        c_bldg_r=0.5):
    return {
        "indomain": {
            "A_raw_depth": {"aggregate": {"all": _agg(r=a_r_id)}},
            "B_global_affine": {"aggregate": {"all": _agg(mae=b_id_mae)}},
            "C_learned_fusion": {"aggregate": {"all": _agg(mae=c_id_mae)}},
        },
        "xcity": {
            "A_raw_depth": {"aggregate": {"all": _agg(r=a_r_id)}},
            "B_global_affine": {"aggregate": {"all": _agg(mae=b_x_mae)}},
            "C_learned_fusion": {"aggregate": {
                "all": _agg(mae=c_x_mae, rmse=c_x_rmse),
                "building": _agg(r=c_bldg_r)}},
        },
    }


def test_go():
    r = _mk(a_r_id=0.6, b_x_mae=3.0, c_x_mae=2.4, c_x_rmse=3.5,
            c_id_mae=2.0, b_id_mae=2.5)
    d = decide(r, evidence_valid=True)
    assert d["verdict"] == "GO", d


def test_modify_affine_as_good():
    # learned head fails to beat affine cross-city -> MODIFY, not GO
    r = _mk(a_r_id=0.6, b_x_mae=3.0, c_x_mae=2.95, c_x_rmse=3.5,
            c_id_mae=2.0, b_id_mae=2.5)
    d = decide(r, evidence_valid=True)
    assert d["verdict"] == "MODIFY", d


def test_abandon_no_signal():
    # raw depth carries ~no monotone height signal -> ABANDON
    r = _mk(a_r_id=0.05, b_x_mae=3.0, c_x_mae=2.4, c_x_rmse=3.5,
            c_id_mae=2.0, b_id_mae=2.5)
    d = decide(r, evidence_valid=True)
    assert d["verdict"] == "ABANDON", d


def test_modify_bad_gen_gap():
    # beats affine but catastrophic generalization gap -> MODIFY (guardrail)
    r = _mk(a_r_id=0.6, b_x_mae=8.0, c_x_mae=6.0, c_x_rmse=9.0,
            c_id_mae=2.0, b_id_mae=2.5)  # xcity/indomain = 3.0 > 2.0
    d = decide(r, evidence_valid=True)
    assert d["verdict"] == "MODIFY", d


def test_inconclusive_missing_c():
    r = _mk(a_r_id=0.6, b_x_mae=3.0, c_x_mae=2.4, c_x_rmse=3.5,
            c_id_mae=2.0, b_id_mae=2.5)
    del r["xcity"]["C_learned_fusion"]   # torch absent scenario
    d = decide(r, evidence_valid=True)
    assert d["verdict"] == "INCONCLUSIVE", d


def test_inconclusive_synthetic():
    r = _mk(a_r_id=0.6, b_x_mae=3.0, c_x_mae=2.4, c_x_rmse=3.5,
            c_id_mae=2.0, b_id_mae=2.5)
    d = decide(r, evidence_valid=False)  # synthetic/fake -> never a real verdict
    assert d["verdict"] == "INCONCLUSIVE", d


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nall {len(fns)} decision tests passed.")
