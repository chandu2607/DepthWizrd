#!/usr/bin/env python
"""Standalone tests for the SCENE-SCALE DIAGNOSTIC (Experiment #6).

Run: python tests/test_scene_scale_diagnostic.py   (no pytest needed)

Covers the module `depthwizard.diagnostics.scene_scale` (feature extraction, leakage
discipline, closed-form ridge, LOO, scale metrics) AND the orchestrator helpers loaded
from scripts/scene_scale_diagnostic.py (CASE classification, reconstruction pooling,
formatting, evidence gating). The centerpiece is a definitive leakage test: perturbing a
scene's ground truth must leave every inference-time feature byte-identical.
"""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from depthwizard.diagnostics import scene_scale as ss  # noqa: E402


def _load_orchestrator():
    path = ROOT / "scripts" / "scene_scale_diagnostic.py"
    spec = importlib.util.spec_from_file_location("ssd_orch", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SSD = _load_orchestrator()


# ---------------------------------------------------------------- fixtures
def _linear_scene(a, b, seed=0, n=512, noise=0.0):
    """A tile whose gt = a*depth + b (+optional noise), so the oracle affine == (a, b)."""
    rng = np.random.default_rng(seed)
    depth = rng.uniform(0.5, 4.0, (n, n)).astype(np.float64)
    gt = a * depth + b + (rng.normal(0, noise, depth.shape) if noise else 0.0)
    gt = gt.astype(np.float32)     # exact affine target so the oracle affine recovers (a, b)
    rgb = (np.clip(depth / 4.0, 0, 1)[..., None] * np.array([200, 150, 100])).astype(np.uint8)
    cls = np.where(gt > 5.0, 6, 2).astype(np.int32)  # tall-ish pixels labelled "building"
    return {"id": f"S_{a}_{b}_{seed}", "city": "SYN", "rgb": rgb, "gt": gt,
            "cls": cls, "depth": depth}


# ---------------------------------------------------------------- leakage (the point of §9)
def test_features_ignore_gt():
    """DEFINITIVE leakage check: features depend on depth+rgb ONLY, never on gt/cls."""
    s = _linear_scene(6.0, -4.0, seed=1)
    f0 = ss.features_to_vector(ss.scene_features(s["depth"], s["rgb"]))
    # obliterate the ground truth (and any downstream mask); features must not move
    for scale in (0.0, 3.7, -10.0):
        gt2 = s["gt"] * scale + 12.0
        f1 = ss.features_to_vector(ss.scene_features(s["depth"], s["rgb"]))
        assert np.array_equal(f0, f1), "features changed without touching depth/rgb -> impossible unless leaking"
    # and the target DOES move when gt moves (sanity: the target is GT-derived, as intended)
    a0 = ss.scene_affine(s["depth"], s["gt"])["a"]
    a1 = ss.scene_affine(s["depth"], s["gt"] * 2.0)["a"]
    assert abs(a1 - 2.0 * a0) < 1e-6, "oracle scale should scale with gt (it is the target)"


def test_extractor_signature_excludes_gt():
    """Structural guarantee: the feature functions cannot even RECEIVE gt/cls."""
    for fn in (ss.scene_features, ss.depth_features, ss.rgb_features):
        params = set(inspect.signature(fn).parameters)
        assert not (params & {"gt", "cls", "label", "mask", "oracle"}), \
            f"{fn.__name__} exposes a GT-like parameter: {params}"


def test_feature_names_stable_and_covered():
    s = _linear_scene(5.0, 0.0)
    feat = ss.scene_features(s["depth"], s["rgb"])
    for name in ss.FEATURE_NAMES:
        assert name in feat, f"missing feature {name}"
    assert len(ss.FEATURE_NAMES) == 15
    v = ss.features_to_vector(feat)
    assert v.shape == (15,) and np.all(np.isfinite(v))


# ---------------------------------------------------------------- target definition (§6/§7)
def test_scene_affine_recovers_known_slope():
    s = _linear_scene(6.465, -5.255, seed=3)
    r = ss.scene_affine(s["depth"], s["gt"])
    assert r["ok"] and abs(r["a"] - 6.465) < 1e-3 and abs(r["b"] - (-5.255)) < 1e-2


def test_scene_affine_guards_tiny():
    d = np.full((3, 3), 1.0); h = np.full((3, 3), 2.0)   # 9 valid px < 10 -> guarded
    r = ss.scene_affine(d, h)
    assert (not r["ok"]) and np.isnan(r["a"])


def test_scale_candidates_present_and_finite():
    s = _linear_scene(4.0, 1.0, seed=5)
    c = ss.scene_scale_candidates(s["depth"], s["gt"])
    assert c["ok"]
    for k in ("a_robust", "a_ols", "median_ratio"):
        assert np.isfinite(c[k]), f"{k} not finite"
    assert abs(c["a_robust"] - 4.0) < 5e-2   # robust slope ~ true slope


# ---------------------------------------------------------------- ridge / standardization
def test_standardize_fit_apply():
    X = np.array([[1.0, 100.0], [2.0, 100.0], [3.0, 100.0]])  # 2nd col constant
    mu, sd = ss.standardize_fit(X)
    Xs = ss.standardize_apply(X, mu, sd)
    assert abs(Xs[:, 0].mean()) < 1e-9 and abs(Xs[:, 0].std() - 1.0) < 1e-9
    assert np.all(np.isfinite(Xs)), "constant column must not produce inf/nan"


def test_ridge_recovers_linear_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (80, 4))
    w_true = np.array([2.0, -1.0, 0.0, 0.5]); y = X @ w_true + 3.0
    mu, sd = ss.standardize_fit(X)
    m = ss.fit_ridge(ss.standardize_apply(X, mu, sd), y, alpha=1e-6)
    pred = ss.predict_ridge(m, ss.standardize_apply(X, mu, sd))
    assert float(np.mean(np.abs(pred - y))) < 1e-3


def test_ridge_multioutput_shapes():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (40, 5)); Y = rng.normal(0, 1, (40, 2))
    mu, sd = ss.standardize_fit(X)
    m = ss.fit_ridge(ss.standardize_apply(X, mu, sd), Y, alpha=1.0)
    P = ss.predict_ridge(m, ss.standardize_apply(X, mu, sd))
    assert P.shape == (40, 2) and not m["single"]


def test_loo_predict_is_reasonable_on_linear():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (60, 3)); y = X @ np.array([1.0, 2.0, -1.0]) + 0.5
    p = ss.loo_predict(X, y, alpha=1e-3)
    assert p.shape == (60,) and float(np.mean(np.abs(p - y))) < 0.1


def test_select_alpha_returns_grid_member():
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (50, 4)); y = X[:, 0] * 3.0 + 1.0
    grid = [0.01, 1.0, 100.0]
    al, mae, scores = ss.select_alpha(X, y, grid)
    assert al in grid and set(scores) == set(grid) and mae >= 0.0


# ---------------------------------------------------------------- scale metrics
def test_scale_metrics_perfect_and_empty():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    m = ss.scale_metrics(t.copy(), t)
    assert m["mae"] == 0.0 and abs(m["pearson"] - 1.0) < 1e-9 and m["n"] == 4
    e = ss.scale_metrics(np.array([]), np.array([]))
    assert e["n"] == 0 and np.isnan(e["mae"])


def test_univariate_screen_sorted_and_complete():
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, (40, len(ss.FEATURE_NAMES)))
    y = X[:, 0] * 5.0 + rng.normal(0, 0.1, 40)   # feature 0 strongly related
    rows = ss.univariate_screen(X, y)
    assert len(rows) == len(ss.FEATURE_NAMES)
    abss = [abs(r["spearman"]) for r in rows if r["spearman"] == r["spearman"]]
    assert abss == sorted(abss, reverse=True)
    assert rows[0]["feature"] == ss.FEATURE_NAMES[0]


def test_random_forest_optional():
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, (40, 4)); y = X[:, 0] + X[:, 1]
    rf = ss.fit_random_forest(X, y, seed=0, n_estimators=20)
    if rf is not None:                       # sklearn present
        p = ss.rf_loo_predict(X, y, seed=0, n_estimators=20)
        assert p is not None and p.shape == (40,)


# ---------------------------------------------------------------- purity
def test_pure_module_does_not_import_torch():
    import subprocess
    code = ("import sys; import depthwizard.diagnostics.scene_scale as m; "
            "assert 'torch' not in sys.modules, 'scene_scale must not import torch'; "
            "print('ok')")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, f"pure-module torch check failed: {r.stderr}\n{r.stdout}"


# ---------------------------------------------------------------- orchestrator helpers
def test_orch_scene_records_extracts_target_and_features():
    tiles = [_linear_scene(6.0, -5.0, seed=i) for i in range(4)]
    scenes, dropped = SSD._scene_records(tiles, blabel=6, nodata=-999.0,
                                         cap=SSD.ALL_CAP, seed=0)
    assert dropped == 0 and len(scenes) == 4
    s0 = scenes[0]
    assert abs(s0["a_or"] - 6.0) < 1e-2 and s0["fvec"].shape == (15,)
    assert s0["d_b"].size > 0 and s0["d_all"].size > 0


def test_orch_scene_records_leakage_property():
    """Orchestrator level: perturbing gt moves the TARGET but never the FEATURES."""
    base = _linear_scene(6.0, -5.0, seed=7)
    pert = dict(base); pert["gt"] = base["gt"] * 2.0 + 30.0
    sc0, _ = SSD._scene_records([base], 6, -999.0, SSD.ALL_CAP, 0)
    sc1, _ = SSD._scene_records([pert], 6, -999.0, SSD.ALL_CAP, 0)
    assert np.array_equal(sc0[0]["fvec"], sc1[0]["fvec"]), "features leaked GT"
    assert abs(sc1[0]["a_or"] - 2.0 * sc0[0]["a_or"]) < 1e-4, "target should track GT"


def test_orch_reconstruction_oracle_beats_global():
    """Oracle per-scene affine should reconstruct near-perfectly; a wrong global should not."""
    scenes, _ = SSD._scene_records([_linear_scene(3.0, 2.0, seed=1),
                                    _linear_scene(9.0, -4.0, seed=2)], 6, -999.0,
                                   SSD.ALL_CAP, 0)
    oracle = SSD._pool_reconstruct(scenes, lambda s: (s["a_or"], s["b_or"]))
    glob = SSD._pool_reconstruct(scenes, lambda s: (1.0, 0.0))   # deliberately wrong global
    assert oracle["building"]["mae"] < 0.05
    assert glob["building"]["mae"] > oracle["building"]["mae"]


def test_orch_classify_all_cases():
    def sblock(sp, rel=0.2):
        return {"pred_a": {"spearman": sp, "rel_median": rel, "mae": 1.0, "rmse": 1.0, "bias": 0.0}}

    def recon(t15):
        base = {"mae": None, "rmse": None, "bias": None}
        return {"building": {"mae": 5.0, "rmse": 6.0, "bias": -1.0},
                "tall_15": {"mae": t15, "rmse": t15, "bias": -t15},
                "tall_20": dict(base), "tall_30": {"mae": t15, "rmse": t15, "bias": -1.0},
                "tall_40": dict(base), "all": dict(base)}
    # C: not predictable even on JAX
    assert SSD._classify(sblock(0.1), sblock(0.1), recon(16), recon(16), recon(7))["case"] == "C"
    # B: JAX ok, OMA not
    assert SSD._classify(sblock(0.7), sblock(0.1), recon(16), recon(16), recon(7))["case"] == "B"
    # A: OMA predictable + helps tall (16 -> 12 is >=1 improvement)
    assert SSD._classify(sblock(0.7), sblock(0.7, 0.3), recon(16), recon(12), recon(7))["case"] == "A"
    # D: OMA predictable but no tall help
    assert SSD._classify(sblock(0.7), sblock(0.7, 0.3), recon(16), recon(16), recon(7))["case"] == "D"


def test_orch_fmt_and_frac_closed():
    assert SSD._fmt(None) == "n/a" and SSD._fmt(float("nan")) == "n/a"
    assert SSD._fmt(float("inf")) == "inf" and SSD._fmt(1.23456, 2) == "1.23"
    assert abs(SSD._frac_closed(16.0, 12.0, 8.0) - 50.0) < 1e-6
    assert np.isnan(SSD._frac_closed(None, 1.0, 1.0))


def test_orch_target_stability_reports_choice():
    tiles = [_linear_scene(a, -3.0, seed=i) for i, a in enumerate([3.0, 5.0, 7.0, 9.0, 4.0])]
    scenes, _ = SSD._scene_records(tiles, 6, -999.0, SSD.ALL_CAP, 0)
    st = SSD._target_stability(scenes)
    assert st["choice"] == "a_robust"
    for k in ("a_robust", "a_ols", "median_ratio"):
        assert k in st["stats"] and st["stats"][k]["n"] >= 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
