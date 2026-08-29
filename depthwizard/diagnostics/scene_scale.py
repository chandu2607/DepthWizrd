"""Per-scene metric-scale diagnostic utilities (Experiment #6). Pure numpy.

Context (the bridge this module tests):
  Experiment #5 (input-signal diagnostic) showed a per-image ORACLE affine
  (h = a*d + b, fit on THAT scene's GT) recovers tall building height, while a
  single GLOBAL affine does not -> the depth->metre SCALE changes scene to scene
  (proposed CASE B). The oracle is NOT deployable: it needs the scene's GT height.

  Experiment #6 asks the deployable question: can the per-scene affine PARAMETERS
  -- the "scene scale" a and offset b -- be PREDICTED from information that will
  actually exist at inference time (the RGB image + the frozen DA-V2 relative
  depth), fit on JAX-train and frozen, without ever using the target scene's GT?

STRICT leakage rule (master prompt §9/§11):
  * The feature extractors take ONLY (depth, rgb). They are structurally incapable
    of touching GT height or the CLS mask -> a feature can never leak the answer.
  * The TARGET (per-scene oracle affine) IS derived from GT -- that is allowed and
    necessary (§7); it is the quantity we are trying to predict, never a feature.
  * The unit of prediction is one SCENE/tile -> one (a, b), never one pixel (§11).

This module stays numpy-only (scikit-learn is optional, used only for the
Method-C nonlinear check) so the whole diagnostic runs off the cached DA-V2
outputs with no torch. Correlation helpers are reused from depth_signal.
"""
from __future__ import annotations

import numpy as np

from .depth_signal import pearson, spearman
from ..models.affine import _fit_affine
from ..metrics.height_metrics import valid_mask

# Ordered inference-time feature set (depth-distribution + spatial + RGB). Every
# entry is computable from a bare RGB image + its frozen DA-V2 depth; NONE uses GT.
FEATURE_NAMES = [
    # --- depth distribution ---
    "depth_mean", "depth_median", "depth_std", "depth_p10", "depth_p90",
    "depth_iqr", "depth_range_p1p99", "depth_skew", "depth_frac_top",
    # --- depth spatial structure ---
    "depth_grad_mean", "depth_grad_std",
    # --- RGB appearance ---
    "rgb_brightness_mean", "rgb_brightness_std", "rgb_edge_mean", "rgb_sat_mean",
]

_EPS = 1e-9


# --------------------------------------------------------------- scene-scale target
def scene_affine(depth, gt, nodata=None, robust=True):
    """Per-scene affine h ~ a*d + b over VALID pixels (the oracle's own params, §6/§7).

    This is exactly the fit inside models.affine.fit_oracle_affine, but returns the
    (a, b) SCALARS -- the per-scene 'oracle scale' target we try to predict. Returns
    ok=False (a=b=nan) when the scene has <10 valid pixels, mirroring the oracle guard.
    """
    d = np.asarray(depth, np.float64)
    h = np.asarray(gt, np.float64)
    m = valid_mask(h, d, nodata=nodata)
    n = int(m.sum())
    if n < 10:
        return {"a": np.nan, "b": np.nan, "n": n, "ok": False}
    a, b = _fit_affine(d[m], h[m], robust=robust)
    return {"a": float(a), "b": float(b), "n": n, "ok": True}


def scene_scale_candidates(depth, gt, nodata=None):
    """Candidate per-scene SCALE statistics for the §6 stability comparison.

    All are GT-derived (this is the target side, not a feature): we inspect which is
    the most stable/meaningful quantity to predict before committing to one.
      * a_robust : robust (trimmed) affine slope  -- the primary candidate
      * a_ols    : plain least-squares affine slope
      * median_ratio : median(h / d) over valid pixels with d > eps (scale-free-ish)
    """
    d = np.asarray(depth, np.float64)
    h = np.asarray(gt, np.float64)
    m = valid_mask(h, d, nodata=nodata)
    if int(m.sum()) < 10:
        return {"a_robust": np.nan, "a_ols": np.nan, "median_ratio": np.nan, "ok": False}
    dm, hm = d[m], h[m]
    a_rob, _ = _fit_affine(dm, hm, robust=True)
    a_ols, _ = _fit_affine(dm, hm, robust=False)
    dr = dm[np.abs(dm) > 1e-3]
    hr = hm[np.abs(dm) > 1e-3]
    ratio = float(np.median(hr / dr)) if dr.size else np.nan
    return {"a_robust": float(a_rob), "a_ols": float(a_ols),
            "median_ratio": ratio, "ok": True}


# --------------------------------------------------------------- inference-time features
def _finite(x):
    x = np.asarray(x, np.float64).ravel()
    return x[np.isfinite(x)]


def _skew(x):
    if x.size < 3:
        return 0.0
    mu = x.mean()
    sd = x.std()
    if sd < _EPS:
        return 0.0
    return float(np.mean(((x - mu) / sd) ** 3))


def depth_features(depth):
    """Depth-distribution + spatial-structure features from the frozen relative depth.

    Uses ONLY the depth map (no GT, no CLS). The relative depth is scale/shift-
    ambiguous, so these describe the SHAPE of the depth distribution / structure of
    the scene -- exactly what is available for a brand-new image.
    """
    D = np.asarray(depth, np.float64)
    d = _finite(D)
    if d.size == 0:
        return {k: 0.0 for k in FEATURE_NAMES if k.startswith("depth_")}
    p1, p10, p25, p50, p75, p90, p99 = np.percentile(d, [1, 10, 25, 50, 75, 90, 99])
    # fraction of pixels in the top quartile of the scene's robust [p1,p99] depth
    # span -- a footprint proxy for "how much of the scene sits at large depth".
    span_hi = p1 + 0.75 * (p99 - p1)
    frac_top = float(np.mean(d > span_hi))
    # spatial gradient magnitude (edge/structure strength); nan-safe via nan->median
    Dg = np.where(np.isfinite(D), D, np.nanmedian(d))
    gy, gx = np.gradient(Dg)
    gmag = np.hypot(gx, gy).ravel()
    return {
        "depth_mean": float(d.mean()),
        "depth_median": float(p50),
        "depth_std": float(d.std()),
        "depth_p10": float(p10),
        "depth_p90": float(p90),
        "depth_iqr": float(p75 - p25),
        "depth_range_p1p99": float(p99 - p1),
        "depth_skew": _skew(d),
        "depth_frac_top": frac_top,
        "depth_grad_mean": float(gmag.mean()),
        "depth_grad_std": float(gmag.std()),
    }


def rgb_features(rgb):
    """Lightweight, interpretable RGB appearance features (§5). ONLY the RGB image.

    Kept deliberately simple (brightness, texture/edge density, saturation) -- no
    giant pretrained vision model (§5). Accepts HxWx3 uint8/float; grayscale is
    broadcast to 3 channels by the loader so this always sees 3 channels.
    """
    a = np.asarray(rgb, np.float64)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    a = a[..., :3]
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gy, gx = np.gradient(gray)
    edge = np.hypot(gx, gy)
    mx = a.max(axis=-1)
    mn = a.min(axis=-1)
    sat = (mx - mn) / (mx + _EPS)      # per-pixel saturation in [0,1]
    return {
        "rgb_brightness_mean": float(gray.mean()),
        "rgb_brightness_std": float(gray.std()),
        "rgb_edge_mean": float(edge.mean()),
        "rgb_sat_mean": float(sat.mean()),
    }


def scene_features(depth, rgb):
    """The inference-time feature dict for ONE scene -- depth + RGB only (no GT/CLS).

    Signature deliberately excludes gt/cls: leakage of the target into the features
    is structurally impossible here (master prompt §9).
    """
    out = {}
    out.update(depth_features(depth))
    out.update(rgb_features(rgb))
    return out


def features_to_vector(feat, names=FEATURE_NAMES):
    return np.array([float(feat.get(n, 0.0)) for n in names], np.float64)


def stack_features(feat_dicts, names=FEATURE_NAMES):
    """(list[dict]) -> (n_scenes, n_features) matrix in canonical FEATURE_NAMES order."""
    if not feat_dicts:
        return np.zeros((0, len(names)), np.float64)
    return np.stack([features_to_vector(f, names) for f in feat_dicts], axis=0)


# --------------------------------------------------------------- standardize + ridge
def standardize_fit(X):
    X = np.asarray(X, np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)     # constant columns -> contribute nothing
    return mu, sd


def standardize_apply(X, mu, sd):
    return (np.asarray(X, np.float64) - mu) / sd


def fit_ridge(Xs, Y, alpha):
    """Closed-form ridge with an UNPENALIZED intercept, single- or multi-output.

    Xs must already be standardized. The intercept is handled by centering Y (so it
    is never shrunk toward 0). Deterministic -> the frozen predictor is reproducible.
    """
    Xs = np.asarray(Xs, np.float64)
    Y = np.asarray(Y, np.float64)
    single = (Y.ndim == 1)
    Ym = Y[:, None] if single else Y
    ymean = Ym.mean(axis=0)
    Yc = Ym - ymean
    p = Xs.shape[1]
    G = Xs.T @ Xs + alpha * np.eye(p)
    W = np.linalg.solve(G, Xs.T @ Yc)          # (p, k)
    return {"W": W, "ymean": ymean, "single": single}


def predict_ridge(model, Xs):
    P = np.asarray(Xs, np.float64) @ model["W"] + model["ymean"]
    return P[:, 0] if model["single"] else P


def loo_predict(Xraw, Y, alpha):
    """Honest leave-one-out predictions on the TRAIN scenes (n small -> refit N times).

    Each fold re-standardizes on the n-1 training rows (the held-out scene never
    influences its own scaler or fit), so LOO is a leakage-free estimate of held-out
    performance -- more stable than the 18-scene JAX-val set alone.
    """
    Xraw = np.asarray(Xraw, np.float64)
    Y = np.asarray(Y, np.float64)
    n = Xraw.shape[0]
    single = (Y.ndim == 1)
    Ym = Y[:, None] if single else Y
    preds = np.zeros_like(Ym, np.float64)
    idx = np.arange(n)
    for i in range(n):
        tr = idx != i
        mu, sd = standardize_fit(Xraw[tr])
        m = fit_ridge(standardize_apply(Xraw[tr], mu, sd), Ym[tr], alpha)
        preds[i] = predict_ridge(m, standardize_apply(Xraw[i:i + 1], mu, sd))
    return preds[:, 0] if single else preds


def select_alpha(Xraw, y, alphas):
    """Pick ridge alpha by LOO MAE on the TRAIN scale target (uses JAX-train only)."""
    best, best_mae = alphas[0], np.inf
    scores = {}
    for al in alphas:
        p = loo_predict(Xraw, y, al)
        mae = float(np.mean(np.abs(p - y)))
        scores[al] = mae
        if mae < best_mae:
            best_mae, best = mae, al
    return best, best_mae, scores


# --------------------------------------------------------------- scale-prediction metrics
def scale_metrics(pred, true):
    """MAE/RMSE/bias/Pearson/Spearman + median relative error for predicted vs true scale."""
    pred = np.asarray(pred, np.float64)
    true = np.asarray(true, np.float64)
    m = np.isfinite(pred) & np.isfinite(true)
    pred, true = pred[m], true[m]
    n = int(pred.size)
    if n == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan,
                "pearson": np.nan, "spearman": np.nan, "rel_median": np.nan}
    err = pred - true
    rel = np.abs(err) / np.maximum(np.abs(true), 1e-6)
    return {"n": n,
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "bias": float(np.mean(err)),
            "pearson": pearson(pred, true),
            "spearman": spearman(pred, true),
            "rel_median": float(np.median(rel))}


def univariate_screen(Xraw, y, names=FEATURE_NAMES):
    """Per-feature Pearson/Spearman vs the scale target -> which cues carry signal (§25 Q2).

    Robust and interpretable regardless of the multivariate fit; sorted by |Spearman|.
    """
    Xraw = np.asarray(Xraw, np.float64)
    y = np.asarray(y, np.float64)
    rows = []
    for j, nm in enumerate(names):
        rows.append({"feature": nm,
                     "pearson": pearson(Xraw[:, j], y),
                     "spearman": spearman(Xraw[:, j], y)})
    rows.sort(key=lambda r: -(abs(r["spearman"]) if r["spearman"] == r["spearman"] else 0.0))
    return rows


# --------------------------------------------------------------- optional nonlinear (Method C)
def fit_random_forest(Xraw, Y, seed=0, n_estimators=200):
    """Method C: sklearn RandomForest, used ONLY if A/B show signal. None if no sklearn."""
    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception:
        return None
    Y = np.asarray(Y, np.float64)
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=seed,
                               min_samples_leaf=3, n_jobs=1)
    rf.fit(np.asarray(Xraw, np.float64), Y)
    return rf


def rf_loo_predict(Xraw, y, seed=0, n_estimators=200):
    """LOO predictions for the RandomForest scale predictor on train (honest CV)."""
    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception:
        return None
    Xraw = np.asarray(Xraw, np.float64)
    y = np.asarray(y, np.float64)
    n = Xraw.shape[0]
    preds = np.zeros(n, np.float64)
    idx = np.arange(n)
    for i in range(n):
        tr = idx != i
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=seed,
                                   min_samples_leaf=3, n_jobs=1)
        rf.fit(Xraw[tr], y[tr])
        preds[i] = rf.predict(Xraw[i:i + 1])[0]
    return preds
