"""Tests for the INPUT-SIGNAL DIAGNOSTIC utilities (master prompt §23).

Covers ONLY diagnostic utilities -- no model behaviour is touched:
  * correct masking (valid + building pixel selection in the extractor)
  * correct bin assignment (bin_index / bin_spans, half-open + trailing inf)
  * no train/test leakage (disjoint_ids guard)
  * correct mapping fitting (affine / poly / isotonic recover known relationships)
  * correct evaluation (map_metrics MAE/RMSE/bias, order_auc, cohens_d, correlations)
  * reproducibility (seeded subsample / spearman / isotonic are deterministic)
  * the pure module stays torch-free (dependency-light, so it runs off the cache alone)

The pure functions live in depthwizard.diagnostics.depth_signal; the extractor / bin-table /
ordering / CASE-classifier live in scripts/depth_signal_diagnostic.py and are loaded here by
path (scripts is not a package). Everything is numpy-only -> runs without pytest:

    python tests/test_depth_signal_diagnostic.py
"""
import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.diagnostics import depth_signal as ds


def _load_orchestrator():
    """Load scripts/depth_signal_diagnostic.py as a module by path (no package)."""
    p = Path(__file__).resolve().parents[1] / "scripts" / "depth_signal_diagnostic.py"
    spec = importlib.util.spec_from_file_location("dsd_orch", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DSD = _load_orchestrator()


# --------------------------------------------------------------- correlations
def test_pearson_known_values():
    x = np.arange(10, dtype=float)
    assert abs(ds.pearson(x, 2 * x + 1) - 1.0) < 1e-9
    assert abs(ds.pearson(x, -3 * x + 5) + 1.0) < 1e-9
    assert ds.pearson(x, np.ones_like(x)) != ds.pearson(x, np.ones_like(x)) or \
        np.isnan(ds.pearson(x, np.ones_like(x)))  # constant -> nan


def test_pearson_constant_and_short_is_nan():
    assert np.isnan(ds.pearson([1.0], [2.0]))
    assert np.isnan(ds.pearson(np.zeros(5), np.arange(5.0)))  # constant x


def test_spearman_rank_monotone_but_nonlinear():
    x = np.linspace(0.1, 3.0, 200)
    y = np.exp(x)  # strictly monotone, very nonlinear
    s = ds.spearman(x, y)
    assert s > 0.999, s               # rank-perfect
    assert ds.pearson(x, y) < s + 1e-9  # linear r is lower than rank r here


def test_rankdata_handles_ties():
    r = ds._rankdata([10, 10, 20, 30])
    # first two tie for ranks 1,2 -> average 1.5 each
    assert np.allclose(r, [1.5, 1.5, 3.0, 4.0]), r


# --------------------------------------------------------------- binning
def test_bin_spans_has_trailing_inf():
    spans = ds.bin_spans([0, 5, 10])
    assert spans == [(0.0, 5.0), (5.0, 10.0), (10.0, float("inf"))]


def test_bin_index_half_open_and_clipping():
    edges = [0, 5, 10, 15]
    idx = ds.bin_index([0.0, 4.9, 5.0, 14.9, 15.0, 99.0, -3.0], edges)
    # [0,5)=0 ; 5.0 -> bin1 ; [10,15)->? 14.9->2 ; 15.0 and above -> last(3) ; negative -> 0
    assert list(idx) == [0, 0, 1, 2, 3, 3, 0], list(idx)


def test_bin_index_matches_spans_length():
    edges = [0, 5, 10, 15, 20, 30, 40]
    spans = ds.bin_spans(edges)
    idx = ds.bin_index(np.array([1, 7, 12, 17, 25, 35, 50], float), edges)
    assert idx.max() < len(spans) and idx.min() >= 0


# --------------------------------------------------------------- mapping fitting
def test_fit_affine_recovers_line():
    d = np.linspace(0, 1, 500)
    h = 6.4 * d - 5.2
    a, b = ds.fit_affine(d, h, robust=False)
    assert abs(a - 6.4) < 1e-6 and abs(b + 5.2) < 1e-6, (a, b)


def test_apply_affine_matches_manual():
    d = np.array([0.0, 1.0, 2.0])
    assert np.allclose(ds.apply_affine((3.0, -1.0), d), [-1.0, 2.0, 5.0])


def test_fit_poly_recovers_quadratic():
    d = np.linspace(-2, 2, 400)
    h = 2.0 * d ** 2 - 0.5 * d + 1.0
    c = ds.fit_poly(d, h, degree=2)
    assert np.allclose(c, [2.0, -0.5, 1.0], atol=1e-6), c


def test_fit_isotonic_monotone_and_recovers():
    d = np.linspace(0, 10, 400)
    h = np.sqrt(d) * 3.0  # monotone increasing
    m = ds.fit_isotonic(d, h, seed=0)
    ys = m.y
    assert np.all(np.diff(ys) >= -1e-9), "isotonic output not non-decreasing"
    # prediction tracks the trend and is bounded by the fitted range
    pred = m.predict(np.array([0.0, 4.0, 9.0]))
    assert pred[0] <= pred[1] <= pred[2]
    assert pred.max() <= ys.max() + 1e-6 and pred.min() >= ys.min() - 1e-6


def test_isotonic_clips_out_of_range():
    d = np.linspace(0, 1, 100)
    m = ds.fit_isotonic(d, 5 * d, seed=0)
    # far outside the fitted domain -> clipped to end values, never extrapolated
    assert m.predict([-100.0])[0] == m.y[0]
    assert m.predict([100.0])[0] == m.y[-1]


def test_isotonic_as_dict_json_serialisable():
    import json
    m = ds.fit_isotonic(np.linspace(0, 1, 50), np.linspace(0, 2, 50), seed=0)
    js = json.dumps(m.as_dict())
    assert '"x"' in js and '"y"' in js


# --------------------------------------------------------------- separability
def test_order_auc_perfect_identical_reversed():
    low = np.zeros(500)
    high = np.ones(500)
    assert abs(ds.order_auc(low, high)[0] - 1.0) < 1e-9      # high strictly above low
    assert abs(ds.order_auc(high, low)[0] - 0.0) < 1e-9      # reversed
    a, _, _ = ds.order_auc(np.arange(500.0), np.arange(500.0))
    assert abs(a - 0.5) < 0.05                                # identical -> ~0.5


def test_order_auc_seeded_reproducible():
    rng = np.random.default_rng(0)
    lo = rng.normal(0, 1, 500_00)
    hi = rng.normal(1, 1, 500_00)
    a1 = ds.order_auc(lo, hi, max_each=1000, seed=7)[0]
    a2 = ds.order_auc(lo, hi, max_each=1000, seed=7)[0]
    assert a1 == a2, (a1, a2)


def test_cohens_d_sign_and_zero():
    a = np.random.default_rng(0).normal(0, 1, 1000)
    b = np.random.default_rng(1).normal(3, 1, 1000)
    assert ds.cohens_d(a, b) > 2.0        # b well above a
    assert ds.cohens_d(b, a) < -2.0
    assert np.isnan(ds.cohens_d(np.ones(10), np.ones(10)))  # zero pooled sd


# --------------------------------------------------------------- evaluation
def test_map_metrics_known():
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    gt = np.array([1.0, 2.0, 2.0, 6.0])  # errors: 0,0,+1,-2
    m = ds.map_metrics(pred, gt)
    assert m["n"] == 4
    assert abs(m["mae"] - 0.75) < 1e-9
    assert abs(m["rmse"] - np.sqrt((0 + 0 + 1 + 4) / 4)) < 1e-9
    assert abs(m["bias"] - (-0.25)) < 1e-9


def test_map_metrics_filters_nonfinite_and_mask():
    pred = np.array([1.0, np.nan, 3.0, 4.0])
    gt = np.array([1.0, 2.0, np.inf, 6.0])
    m = ds.map_metrics(pred, gt)
    assert m["n"] == 2                    # only idx 0 and 3 finite in both
    mask = np.array([True, True, True, False])
    m2 = ds.map_metrics(pred, gt, mask=mask)
    assert m2["n"] == 1                   # mask drops idx3, nan/inf drop 1&2


def test_map_metrics_empty_is_nan():
    m = ds.map_metrics(np.array([]), np.array([]))
    assert m["n"] == 0 and np.isnan(m["mae"]) and np.isnan(m["rmse"])


# --------------------------------------------------------------- reproducibility / guards
def test_subsample_deterministic_and_capped():
    a = np.arange(1000)
    s1 = ds.subsample(a, 100, seed=3)
    s2 = ds.subsample(a, 100, seed=3)
    assert s1.shape[0] == 100 and np.array_equal(s1, s2)
    assert np.array_equal(ds.subsample(a, 5000, seed=3), a)  # no-op when k >= n


def test_disjoint_ids():
    assert ds.disjoint_ids(["a", "b"], ["c", "d"]) is True
    assert ds.disjoint_ids(["a", "b"], ["b", "c"]) is False


def test_pure_module_does_not_import_torch():
    """The diagnostic must be able to run off the cache alone (§5) -> the pure numeric
    module must not drag in torch/transformers."""
    if "torch" in sys.modules:
        print("  (torch already present in this process; skipping strict absence check)")
        return
    importlib.reload(ds)
    assert "torch" not in sys.modules, "depth_signal pulled in torch"


# --------------------------------------------------------------- extractor (masking)
def _tile(sid, depth, gt, cls):
    return {"id": sid, "depth": np.asarray(depth, float),
            "gt": np.asarray(gt, float), "cls": np.asarray(cls, int), "city": "T"}


def test_extract_masks_valid_and_buildings():
    # 3x3 tile: a 2-pixel building (cls==6), one NaN (nodata) ground pixel.
    depth = np.array([[0.1, 0.2, 0.3],
                      [0.4, 0.9, 0.95],
                      [0.5, 0.6, 0.7]])
    gt = np.array([[0.0, 0.0, 0.0],
                   [np.nan, 20.0, 22.0],   # NaN = nodata -> must be dropped
                   [0.0, 0.0, 0.0]])
    cls = np.array([[2, 2, 2],
                    [2, 6, 6],             # two building pixels
                    [2, 2, 2]])
    dom = DSD._extract([_tile("a", depth, gt, cls)], blabel=6, nodata=-999.0,
                       cap=100, seed=0)
    # building pool: exactly the two cls==6 valid pixels, heights 20 & 22
    assert dom["d_b"].size == 2 and set(np.round(dom["h_b"], 3)) == {20.0, 22.0}
    assert set(np.round(dom["d_b"], 3)) == {0.9, 0.95}
    # all-valid pool: 9 - 1 NaN = 8 pixels
    assert dom["h_all"].size == 8 and np.isfinite(dom["h_all"]).all()
    # oracle predictions are produced per building pixel (same count as building GT)
    assert dom["o_b"].size == dom["h_b"].size == 2


def test_extract_no_building_pixels_is_safe():
    depth = np.array([[0.1, 0.2], [0.3, 0.4]])
    gt = np.array([[0.0, 1.0], [2.0, 3.0]])
    cls = np.full((2, 2), 2, int)          # no buildings
    dom = DSD._extract([_tile("a", depth, gt, cls)], blabel=6, nodata=-999.0,
                       cap=100, seed=0)
    assert dom["d_b"].size == 0 and dom["h_b"].size == 0
    assert dom["d_all"].size == 4


def test_merge_concatenates_pools():
    t1 = _tile("a", [[0.5]], [[10.0]], [[6]])
    t2 = _tile("b", [[0.6]], [[20.0]], [[6]])
    d1 = DSD._extract([t1], 6, -999.0, 100, 0)
    d2 = DSD._extract([t2], 6, -999.0, 100, 0)
    m = DSD._merge(d1, d2)
    assert m["d_b"].size == 2 and set(m["ids"]) == {"a", "b"}


# --------------------------------------------------------------- bin table / ordering
def test_bin_table_counts_and_mapping_error():
    # buildings at heights 3 (bin 0-5) and 12 (bin 10-15); identity-ish affine
    d_b = np.array([3.0, 12.0])
    h_b = np.array([3.0, 12.0])
    dom = {"d_b": d_b, "h_b": h_b}
    rows = DSD._bin_table(dom, ab=(1.0, 0.0), edges=DSD.DIAG_EDGES, seed=0)
    b0 = next(r for r in rows if r["lo"] == 0.0)
    b1015 = next(r for r in rows if r["lo"] == 10.0)
    assert b0["n"] == 1 and abs(b0["map_mae"]) < 1e-9      # perfect map -> 0 error
    assert b1015["n"] == 1 and abs(b1015["map_mae"]) < 1e-9
    empty = next(r for r in rows if r["lo"] == 20.0)
    assert empty["n"] == 0 and empty["map_mae"] is None


def test_ordering_monotone_depth_tracks_height():
    # depth increases with height across bins -> monotone, high tall AUC
    rng = np.random.default_rng(0)
    heights, depths = [], []
    for base in (2, 7, 12, 17, 25, 35, 45):
        heights.append(np.full(200, base, float))
        depths.append(base * 0.1 + rng.normal(0, 0.01, 200))  # tight, monotone
    dom = {"d_b": np.concatenate(depths), "h_b": np.concatenate(heights)}
    o = DSD._ordering(dom, DSD.DIAG_EDGES, seed=0)
    assert o["mono_fraction"] == 1.0
    assert o["adjacent_tall_auc_mean"] > 0.95
    assert o["spearman_building"] > 0.95


def test_ordering_no_signal_auc_near_half():
    rng = np.random.default_rng(0)
    # depth independent of height -> ordering AUC ~ 0.5, near-zero spearman
    h = rng.integers(0, 50, 4000).astype(float)
    d = rng.normal(0, 1, 4000)
    o = DSD._ordering({"d_b": d, "h_b": h}, DSD.DIAG_EDGES, seed=0)
    assert abs(o["all_pairs_auc_mean"] - 0.5) < 0.05
    assert abs(o["spearman_building"]) < 0.1


# --------------------------------------------------------------- CASE classifier
def _mk(spear15, auc_tall, glob_tall_mae, orac_tall_mae):
    tall_only = {"gt_15": {"spearman": spear15, "pearson": spear15, "n": 1000},
                 "gt_20": {"spearman": spear15, "pearson": spear15, "n": 500},
                 "gt_30": {"spearman": spear15, "pearson": spear15, "n": 100}}
    ordering = {"adjacent_tall_auc_mean": auc_tall}
    glob = {"tall_15": {"mae": glob_tall_mae}}
    orac = {"tall_15": {"mae": orac_tall_mae}}
    return tall_only, ordering, glob, orac


def test_classify_case_a_strong_signal():
    c = DSD._classify(*_mk(0.6, 0.82, 5.0, 3.0))
    assert c["case"] == "A", c


def test_classify_case_b_scene_scale_ambiguity():
    # ordering ok (not strong: spearman 0.30 < 0.40, not weak), oracle recovers, big gap
    c = DSD._classify(*_mk(0.30, 0.66, 12.0, 4.0))
    assert c["case"] == "B", c


def test_classify_case_c_weak_tall_signal():
    c = DSD._classify(*_mk(0.05, 0.52, 12.0, 11.0))
    assert c["case"] == "C", c


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
        passed += 1
    print(f"\nall {passed} depth-signal diagnostic tests passed.")
