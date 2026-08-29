"""Pure numeric utilities for the DepthWizard INPUT-SIGNAL DIAGNOSTIC.

These functions answer one question about the FROZEN Depth Anything V2 relative-depth
representation: *does it already contain recoverable tall-building-height information?*
They are deliberately dependency-light (numpy, with scipy/sklearn used where present and
a numpy fallback otherwise) and side-effect-free, so `tests/test_depth_signal_diagnostic.py`
can exercise every one deterministically. All IO, figures and reporting live in
`scripts/depth_signal_diagnostic.py`; NOTHING here trains a model or touches the fusion head.

Key metrics
-----------
- pearson / spearman : linear and rank correlation of depth vs GT height.
- fit_affine / fit_poly / fit_isotonic : the three SIMPLE monotonic mappings h ~ f(depth)
  the diagnostic fits (on JAX-train ONLY) to test whether the signal is *recoverable*.
- order_auc : P(depth_high > depth_low) for random cross-bin pairs (Mann-Whitney U / n_l*n_h).
  This is THE separability metric: 0.5 = the two height bins are indistinguishable in depth,
  1.0 = perfectly ordered. If tall bins score ~0.5, no monotone mapping can recover them.
- cohens_d : standardized mean gap between two depth distributions (bin separation).
- map_metrics : MAE/RMSE/bias/pearson/spearman of a mapping's prediction vs GT.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# correlations
# --------------------------------------------------------------------------- #
def pearson(x, y) -> float:
    """Pearson linear correlation; NaN for <2 points or a constant input."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size != x.size:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(a) -> np.ndarray:
    """Average-rank transform with tie handling (mergesort = stable). numpy fallback."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    ranks = np.empty(a.size, dtype=np.float64)
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0  # 1-based average rank
        i = j + 1
    return ranks


def spearman(x, y, max_n: int = 3_000_000, seed: int = 0) -> float:
    """Spearman rank correlation. Subsamples to `max_n` (seeded) for speed on millions
    of pixels. Uses scipy when available; else average-rank + Pearson (tie-correct)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size != x.size:
        return float("nan")
    if x.size > max_n:
        idx = np.random.default_rng(seed).choice(x.size, max_n, replace=False)
        x, y = x[idx], y[idx]
    try:
        from scipy.stats import spearmanr
        r = spearmanr(x, y).correlation
        return float(r)
    except Exception:
        return pearson(_rankdata(x), _rankdata(y))


# --------------------------------------------------------------------------- #
# simple monotonic mappings  h ~ f(depth)   (fit on TRAIN ONLY by the caller)
# --------------------------------------------------------------------------- #
def fit_affine(d, h, robust: bool = True):
    """Least-squares affine h ~ a*depth + b (optional one-pass 10%-trim robustness).
    Reuses the project's canonical fitter so it matches Baseline B exactly."""
    from ..models.affine import _fit_affine
    return _fit_affine(np.asarray(d, np.float64), np.asarray(h, np.float64), robust=robust)


def apply_affine(ab, d) -> np.ndarray:
    a, b = ab
    return (a * np.asarray(d, np.float64) + b)


def fit_poly(d, h, degree: int = 2, max_fit: int = 3_000_000, seed: int = 0):
    """Low-degree polynomial h ~ f(depth) (degree 2 by default)."""
    d = np.asarray(d, np.float64)
    h = np.asarray(h, np.float64)
    if d.size > max_fit:
        idx = np.random.default_rng(seed).choice(d.size, max_fit, replace=False)
        d, h = d[idx], h[idx]
    return np.polyfit(d, h, degree)


def apply_poly(coeffs, d) -> np.ndarray:
    return np.polyval(coeffs, np.asarray(d, np.float64))


class IsotonicMap:
    """A frozen, monotone-nondecreasing lookup h = f(depth), stored as a small grid so it
    is JSON-serialisable and cheap to apply. `predict` linearly interpolates / clips."""

    def __init__(self, x_grid, y_grid):
        self.x = np.asarray(x_grid, np.float64)
        self.y = np.asarray(y_grid, np.float64)

    def predict(self, d) -> np.ndarray:
        return np.interp(np.asarray(d, np.float64), self.x, self.y,
                         left=self.y[0], right=self.y[-1])

    def as_dict(self) -> dict:
        return {"x": self.x.tolist(), "y": self.y.tolist()}


def fit_isotonic(d, h, n_grid: int = 512, max_fit: int = 2_000_000, seed: int = 0) -> IsotonicMap:
    """Best monotone-nondecreasing mapping (isotonic regression), sampled onto a grid.

    This is the STRONGEST simple mapping the diagnostic tries: if even the best monotone
    function of depth cannot climb the tall tail, the shortfall is missing signal, not a
    wrong functional form. Uses sklearn's PAVA; falls back to a quantile-binned monotone
    (pool-adjacent-violators) if sklearn is unavailable."""
    d = np.asarray(d, np.float64)
    h = np.asarray(h, np.float64)
    if d.size > max_fit:
        idx = np.random.default_rng(seed).choice(d.size, max_fit, replace=False)
        d, h = d[idx], h[idx]
    lo, hi = float(np.min(d)), float(np.max(d))
    if hi <= lo:
        return IsotonicMap([lo, lo + 1e-6], [float(np.mean(h))] * 2)
    xs = np.linspace(lo, hi, n_grid)
    try:
        from sklearn.isotonic import IsotonicRegression
        ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
        ir.fit(d, h)
        ys = np.asarray(ir.predict(xs), np.float64)
    except Exception:
        ys = _binned_isotonic(d, h, xs)
    # guarantee monotone non-decreasing on the grid
    ys = np.maximum.accumulate(ys)
    return IsotonicMap(xs, ys)


def _binned_isotonic(d, h, xs) -> np.ndarray:
    """Fallback: mean h per depth-quantile bin, made monotone via pool-adjacent-violators."""
    q = np.linspace(0, 1, 257)
    edges = np.quantile(d, q)
    edges = np.unique(edges)
    if edges.size < 2:
        return np.full(xs.size, float(np.mean(h)))
    idx = np.clip(np.digitize(d, edges[1:-1], right=False), 0, edges.size - 2)
    means, centers = [], []
    for b in range(edges.size - 1):
        sel = idx == b
        if sel.any():
            means.append(float(np.mean(h[sel])))
            centers.append(0.5 * (edges[b] + edges[b + 1]))
    means = np.asarray(means, np.float64)
    centers = np.asarray(centers, np.float64)
    # PAVA
    y = means.copy()
    w = np.ones_like(y)
    i = 0
    while i < len(y) - 1:
        if y[i] > y[i + 1]:
            new = (y[i] * w[i] + y[i + 1] * w[i + 1]) / (w[i] + w[i + 1])
            y[i] = new
            w[i] += w[i + 1]
            y = np.delete(y, i + 1)
            w = np.delete(w, i + 1)
            centers = np.delete(centers, i + 1)
            if i > 0:
                i -= 1
        else:
            i += 1
    if centers.size < 2:
        return np.full(xs.size, float(means.mean()))
    return np.interp(xs, centers, y)


# --------------------------------------------------------------------------- #
# height binning
# --------------------------------------------------------------------------- #
def bin_spans(edges):
    """Ascending edges -> half-open [lo, hi) spans with a trailing [last, inf) bin."""
    e = [float(x) for x in edges]
    spans = list(zip(e[:-1], e[1:]))
    spans.append((e[-1], float("inf")))
    return spans


def bin_index(h, edges) -> np.ndarray:
    """Bin index in [0, len(edges)-1] for each height (last bin is [edges[-1], inf))."""
    edges = np.asarray(edges, np.float64)
    idx = np.digitize(np.asarray(h, np.float64), edges, right=False) - 1
    return np.clip(idx, 0, len(edges) - 1)


# --------------------------------------------------------------------------- #
# separability: cross-bin ordering AUC and standardized gap
# --------------------------------------------------------------------------- #
def order_auc(x_low, x_high, max_each: int = 200_000, seed: int = 0):
    """P(sample from `x_high` > sample from `x_low`) = Mann-Whitney U / (n_low*n_high).

    0.5 = the two groups' depth distributions are indistinguishable (no usable ordering);
    1.0 = perfectly separated (every tall-bin pixel has larger depth than every lower one).
    Subsamples each group to `max_each` (seeded). Returns (auc, n_low, n_high)."""
    xl = np.asarray(x_low, np.float64)
    xh = np.asarray(x_high, np.float64)
    if xl.size == 0 or xh.size == 0:
        return float("nan"), int(xl.size), int(xh.size)
    rng = np.random.default_rng(seed)
    if xl.size > max_each:
        xl = rng.choice(xl, max_each, replace=False)
    if xh.size > max_each:
        xh = rng.choice(xh, max_each, replace=False)
    try:
        from scipy.stats import mannwhitneyu
        u = mannwhitneyu(xh, xl, alternative="greater").statistic
        auc = float(u / (xh.size * xl.size))
    except Exception:
        allv = np.concatenate([xh, xl])
        r = _rankdata(allv)
        rh = r[:xh.size].sum()
        auc = float((rh - xh.size * (xh.size + 1) / 2.0) / (xh.size * xl.size))
    return auc, int(xl.size), int(xh.size)


def cohens_d(a, b) -> float:
    """Standardized mean difference (mean_b - mean_a)/pooled_sd. |d|>~0.8 = large gap."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.size < 2 or b.size < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = np.sqrt(((a.size - 1) * va + (b.size - 1) * vb) / max(a.size + b.size - 2, 1))
    if sp < 1e-12:
        return float("nan")
    return float((np.mean(b) - np.mean(a)) / sp)


# --------------------------------------------------------------------------- #
# mapping evaluation
# --------------------------------------------------------------------------- #
def map_metrics(pred, gt, mask=None) -> dict:
    """MAE/RMSE/bias(pred-gt)/pearson/spearman over finite (optionally masked) pixels."""
    pred = np.asarray(pred, np.float64)
    gt = np.asarray(gt, np.float64)
    if mask is not None:
        mask = np.asarray(mask, bool)
        pred, gt = pred[mask], gt[mask]
    m = np.isfinite(pred) & np.isfinite(gt)
    pred, gt = pred[m], gt[m]
    n = int(pred.size)
    if n == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"),
                "bias": float("nan"), "pearson": float("nan"), "spearman": float("nan")}
    err = pred - gt
    return {"n": n,
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "bias": float(np.mean(err)),
            "pearson": pearson(pred, gt),
            "spearman": spearman(pred, gt)}


def subsample(a, k: int, seed: int = 0) -> np.ndarray:
    """Deterministic row subsample of a 1-D array to at most k elements."""
    a = np.asarray(a)
    if a.shape[0] <= k:
        return a
    idx = np.random.default_rng(seed).choice(a.shape[0], k, replace=False)
    return a[idx]


def disjoint_ids(fit_ids, eval_ids) -> bool:
    """No-leakage guard: True iff the fit-set and eval-set tile ids are disjoint (§16)."""
    return len(set(fit_ids) & set(eval_ids)) == 0
