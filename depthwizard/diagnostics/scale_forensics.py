"""Pure-numeric utilities for the DepthWizard DATASET / SCALE FORENSICS study (Experiment #7).

This module answers descriptive, measurable questions about how the JAX-train / JAX-val /
OMA-test splits differ -- in image resolution, height distribution, frozen-DA-V2 depth
distribution, per-scene depth->metre oracle slope, scene composition and tall-structure
representation. It is deliberately dependency-light (numpy; scipy.ndimage used for connected
components when present, with a numpy flood-fill fallback) and side-effect-free, so
`tests/test_dataset_scale_forensics.py` can exercise every function deterministically.

NOTHING here trains a model, touches the fusion head / C_log1p / loss / transform / DA-V2, or
recomputes depth. All IO, the Experiment-#6 prediction recovery (Part A), figures and reporting
live in `scripts/dataset_scale_forensics.py`.

Design notes
------------
* "Valid pixel" is defined identically to the rest of the project -- `valid_mask(h, d)` (finite
  GT that is not the nodata sentinel AND finite depth) -- so pixel counts / fractions here match
  Phase-1..6 and the oracle affine domain exactly.
* Exact quantities (valid counts, class fractions, threshold proportions) are computed on FULL
  tiles (cheap boolean means). Distribution SHAPE (percentiles, histograms) is estimated from a
  seeded per-tile subsample pooled across a split, so a 60M-pixel split stays in memory; the cap
  and seed are recorded for reproducibility.
* GSD: these mirror crops carry only a uniform identity ModelTransformation stub and no CRS, so a
  true physical pixel size is UNAVAILABLE. `raw_raster_info` reports what is actually in the file
  and never fabricates a GSD (master prompt §10).
"""
from __future__ import annotations

import os
from collections import deque

import numpy as np

from .depth_signal import pearson, spearman, cohens_d
from ..metrics.height_metrics import valid_mask

DEFAULT_HEIGHT_THRESHOLDS = (5.0, 10.0, 15.0, 20.0, 30.0, 40.0)
DEFAULT_PERCENTILES = (1, 5, 10, 25, 50, 75, 85, 90, 95, 99)
TALL_BANDS = [(15.0, 20.0), (20.0, 30.0), (30.0, 40.0), (40.0, float("inf"))]
GROUND_LABEL = 2


# --------------------------------------------------------------------------- helpers
def finite(x) -> np.ndarray:
    x = np.asarray(x, np.float64).ravel()
    return x[np.isfinite(x)]


def _pctiles(x, ps=DEFAULT_PERCENTILES) -> dict:
    x = finite(x)
    if x.size == 0:
        return {f"p{int(p)}": float("nan") for p in ps}
    vals = np.percentile(x, list(ps))
    return {f"p{int(p)}": float(v) for p, v in zip(ps, vals)}


def _basic(x) -> dict:
    """mean/median/std/min/max + percentiles + n for a 1-D sample (nan-safe)."""
    x = finite(x)
    n = int(x.size)
    if n == 0:
        base = {"n": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan")}
        base.update(_pctiles(x))
        return base
    base = {"n": n, "mean": float(x.mean()), "median": float(np.median(x)),
            "std": float(x.std()), "min": float(x.min()), "max": float(x.max())}
    base.update(_pctiles(x))
    return base


def frac_above(x, thresholds=DEFAULT_HEIGHT_THRESHOLDS) -> dict:
    """Fraction of finite values strictly above each threshold."""
    x = finite(x)
    if x.size == 0:
        return {f">{_tk(t)}m": float("nan") for t in thresholds}
    return {f">{_tk(t)}m": float(np.mean(x > t)) for t in thresholds}


def _tk(t) -> str:
    return str(int(t)) if float(t).is_integer() else str(t)


# --------------------------------------------------------------------------- raw raster metadata (§10/§11)
def raw_raster_info(path: str) -> dict:
    """Read on-disk dimensions / dtype / format / any geo-transform WITHOUT the project resize.

    Reports exactly what the file carries. GeoTIFF ModelTransformation scale is reported as
    `pixel_scale`; `georeferenced` is True only when a CRS/GeoKeyDirectory is present. For the
    JasonXF mirror these are absent (identity stub, scale 1.0), so a physical GSD is UNAVAILABLE
    and is returned as None -- never fabricated (§10).
    """
    info = {"path": str(path), "exists": os.path.exists(path), "shape": None, "dtype": None,
            "format": os.path.splitext(str(path))[1].lower().lstrip("."),
            "pixel_scale": None, "georeferenced": False, "crs": None, "gsd_m": None}
    if not info["exists"]:
        return info
    ext = info["format"]
    if ext in ("tif", "tiff"):
        try:
            import tifffile
            with tifffile.TiffFile(path) as tf:
                p = tf.pages[0]
                shp = tuple(int(s) for s in p.shape)
                info["shape"] = shp
                info["dtype"] = str(p.dtype)
                tagnames = {t.name for t in p.tags.values()}
                # CRS is signalled by a GeoKeyDirectoryTag; without it the file is not georeferenced.
                info["georeferenced"] = "GeoKeyDirectoryTag" in tagnames
                for t in p.tags.values():
                    if t.name in ("ModelTransformationTag", "ModelPixelScaleTag"):
                        v = np.asarray(t.value, float)
                        if t.name == "ModelPixelScaleTag" and v.size >= 2:
                            info["pixel_scale"] = [float(v[0]), float(v[1])]
                        elif t.name == "ModelTransformationTag" and v.size >= 6:
                            M = v.reshape(4, 4)
                            info["pixel_scale"] = [float(np.hypot(M[0, 0], M[1, 0])),
                                                   float(np.hypot(M[0, 1], M[1, 1]))]
                # GSD only if genuinely georeferenced (never invent one from the stub).
                if info["georeferenced"] and info["pixel_scale"]:
                    info["gsd_m"] = float(info["pixel_scale"][0])
            return info
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {e}"
            return info
    # PNG / other -> PIL for size only.
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
            info["shape"] = (int(h), int(w)) + ((len(im.getbands()),) if im.getbands() else ())
            info["dtype"] = {"L": "uint8", "RGB": "uint8", "RGBA": "uint8", "I": "int32",
                             "I;16": "uint16", "F": "float32"}.get(im.mode, im.mode)
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def summarize_raw_infos(infos: list[dict]) -> dict:
    """Aggregate a list of raw_raster_info dicts: unique shapes/dtypes/pixel-scales/georef."""
    def _counts(vals):
        out = {}
        for v in vals:
            k = str(v)
            out[k] = out.get(k, 0) + 1
        return out
    ok = [i for i in infos if i.get("shape") is not None]
    return {"n": len(infos), "n_readable": len(ok),
            "shapes": _counts(i["shape"] for i in ok),
            "dtypes": _counts(i["dtype"] for i in ok),
            "formats": _counts(i["format"] for i in infos),
            "pixel_scales": _counts(i.get("pixel_scale") for i in ok),
            "georeferenced": _counts(i.get("georeferenced") for i in ok),
            "gsd_available": any(i.get("gsd_m") is not None for i in ok)}


# --------------------------------------------------------------------------- height distribution (§12/§18)
def height_stats(h, d, cls=None, blabel=6, nodata=None,
                 thresholds=DEFAULT_HEIGHT_THRESHOLDS) -> dict:
    """Per-tile height distribution over ALL-valid and BUILDING pixels (§12/§18).

    Validity = valid_mask(h, d) (finite non-nodata GT AND finite depth), identical to the rest
    of the project. Ground/building fractions are over valid pixels using the CLS mask.
    Returns exact counts + fractions; the height arrays for pooled distribution shape are the
    caller's job (see pooled_distribution). `frac_above` is computed here on the full tile.
    """
    h = np.asarray(h, np.float64)
    d = np.asarray(d, np.float64)
    vm = valid_mask(h, d, nodata=nodata)
    n_valid = int(vm.sum())
    out = {"n_valid": n_valid}
    if cls is not None:
        c = np.asarray(cls)
        out["ground_frac"] = float(np.mean(c[vm] == GROUND_LABEL)) if n_valid else float("nan")
        out["building_frac"] = float(np.mean(c[vm] == blabel)) if n_valid else float("nan")
        bmask = vm & (c == blabel)
    else:
        out["ground_frac"] = float("nan")
        out["building_frac"] = float("nan")
        bmask = np.zeros_like(vm)
    hv = h[vm]
    hb = h[bmask]
    out["all"] = {**_basic(hv), "frac_above": frac_above(hv, thresholds)}
    out["building"] = {**_basic(hb), "frac_above": frac_above(hb, thresholds)}
    out["n_building"] = int(bmask.sum())
    return out


# --------------------------------------------------------------------------- depth distribution (§13/§19)
def depth_stats(d, mask=None, hist_bins=40, hist_range=None) -> dict:
    """Depth distribution over (optionally masked) finite pixels + a histogram for overlap."""
    d = np.asarray(d, np.float64)
    if mask is not None:
        d = d[np.asarray(mask, bool)]
    dv = finite(d)
    out = _basic(dv)
    if hist_range is None:
        hist_range = (float(dv.min()), float(dv.max())) if dv.size else (0.0, 1.0)
    if hist_range[1] <= hist_range[0]:
        hist_range = (hist_range[0], hist_range[0] + 1e-6)
    counts, edges = np.histogram(dv, bins=hist_bins, range=hist_range)
    out["hist_counts"] = counts.astype(int).tolist()
    out["hist_edges"] = edges.astype(float).tolist()
    return out


def hist_overlap(counts_a, counts_b) -> float:
    """Histogram-intersection overlap in [0,1] for two count vectors on the SAME edges.

    1.0 = identical normalized distributions; 0.0 = disjoint support. Requires equal length.
    """
    a = np.asarray(counts_a, np.float64)
    b = np.asarray(counts_b, np.float64)
    if a.size == 0 or a.size != b.size:
        return float("nan")
    sa, sb = a.sum(), b.sum()
    if sa <= 0 or sb <= 0:
        return float("nan")
    return float(np.minimum(a / sa, b / sb).sum())


def ks_2samp(x, y, max_n=200000, seed=0) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (max CDF gap) in [0,1]; numpy-only, subsampled."""
    x = finite(x)
    y = finite(y)
    if x.size == 0 or y.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    if x.size > max_n:
        x = rng.choice(x, max_n, replace=False)
    if y.size > max_n:
        y = rng.choice(y, max_n, replace=False)
    xs = np.sort(x)
    ys = np.sort(y)
    allv = np.concatenate([xs, ys])
    allv.sort()
    cdf_x = np.searchsorted(xs, allv, side="right") / xs.size
    cdf_y = np.searchsorted(ys, allv, side="right") / ys.size
    return float(np.max(np.abs(cdf_x - cdf_y)))


def tall_band_depth_stats(d, h, cls=None, blabel=6, nodata=None, bands=TALL_BANDS) -> dict:
    """Frozen-DA-V2 depth distribution for BUILDING pixels whose GT height lies in each tall band.

    Answers §19: do tall OMA structures occupy a different DA depth range than tall JAX ones?
    Returns per-band {n, depth mean/median/std/p10/p90}. Building pixels only (falls back to all
    valid pixels if no CLS), within valid_mask(h,d).
    """
    d = np.asarray(d, np.float64)
    h = np.asarray(h, np.float64)
    vm = valid_mask(h, d, nodata=nodata)
    if cls is not None:
        vm = vm & (np.asarray(cls) == blabel)
    out = {}
    for lo, hi in bands:
        sel = vm & (h >= lo) & (h < hi)
        dd = d[sel]
        key = f"{_tk(lo)}_{_tk(hi) if np.isfinite(hi) else 'inf'}"
        if dd.size:
            p10, p50, p90 = np.percentile(dd, [10, 50, 90])
            out[key] = {"n": int(dd.size), "depth_mean": float(dd.mean()),
                        "depth_median": float(p50), "depth_std": float(dd.std()),
                        "depth_p10": float(p10), "depth_p90": float(p90)}
        else:
            out[key] = {"n": 0, "depth_mean": float("nan"), "depth_median": float("nan"),
                        "depth_std": float("nan"), "depth_p10": float("nan"),
                        "depth_p90": float("nan")}
    return out


# --------------------------------------------------------------------------- connected components (§20/§21)
def label_components(mask) -> tuple[np.ndarray, int]:
    """4-connected connected-component labelling. scipy.ndimage.label if present, else flood-fill."""
    mask = np.asarray(mask, bool)
    try:
        from scipy import ndimage
        lab, n = ndimage.label(mask)
        return lab.astype(np.int32), int(n)
    except Exception:
        return _flood_label(mask)


def _flood_label(mask) -> tuple[np.ndarray, int]:
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32)
    n = 0
    for i in range(H):
        for j in range(W):
            if mask[i, j] and lab[i, j] == 0:
                n += 1
                q = deque([(i, j)])
                lab[i, j] = n
                while q:
                    y, x = q.popleft()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and lab[ny, nx] == 0:
                            lab[ny, nx] = n
                            q.append((ny, nx))
    return lab, n


def footprint_stats(bmask, large_px=1000) -> dict:
    """Building-footprint size stats from connected components (§20). Simple, per tile."""
    bmask = np.asarray(bmask, bool)
    total = int(bmask.sum())
    lab, n = label_components(bmask)
    if n == 0:
        return {"building_px": total, "n_components": 0, "largest_px": 0,
                "mean_px": float("nan"), "median_px": float("nan"), "n_large": 0}
    sizes = np.bincount(lab.ravel())[1:]  # drop background label 0
    return {"building_px": total, "n_components": int(n),
            "largest_px": int(sizes.max()), "mean_px": float(sizes.mean()),
            "median_px": float(np.median(sizes)), "n_large": int((sizes >= large_px).sum())}


def border_stats(mask, border=2) -> dict:
    """Edge-clipping stats for a boolean structure mask (§21).

    Returns fraction of mask pixels within `border` px of a tile edge, and the fraction of
    connected components that touch the edge (a proxy for clipped/partial structures).
    """
    mask = np.asarray(mask, bool)
    total = int(mask.sum())
    if total == 0:
        return {"px": 0, "border_px_frac": float("nan"), "n_components": 0,
                "n_border_components": 0, "border_component_frac": float("nan")}
    H, W = mask.shape
    edge = np.zeros_like(mask)
    edge[:border, :] = edge[-border:, :] = edge[:, :border] = edge[:, -border:] = True
    border_px = int((mask & edge).sum())
    lab, n = label_components(mask)
    touch = 0
    if n:
        edge_labels = set(np.unique(lab[edge & mask]).tolist()) - {0}
        touch = len(edge_labels)
    return {"px": total, "border_px_frac": border_px / total, "n_components": int(n),
            "n_border_components": int(touch),
            "border_component_frac": (touch / n) if n else float("nan")}


# --------------------------------------------------------------------------- split-shift quantification (§15/§23)
def split_shift(a, b) -> dict:
    """Quantify the shift between two per-scene samples a (e.g. JAX) and b (e.g. OMA) (§15).

    Reports central tendencies, absolute + ratio differences, standardized effect size (Cohen's d)
    and the KS statistic. `ratio` = mean(a)/mean(b) (JAX-over-OMA when called that way).
    """
    a = finite(a)
    b = finite(b)
    ma = float(a.mean()) if a.size else float("nan")
    mb = float(b.mean()) if b.size else float("nan")
    mda = float(np.median(a)) if a.size else float("nan")
    mdb = float(np.median(b)) if b.size else float("nan")
    return {"n_a": int(a.size), "n_b": int(b.size),
            "mean_a": ma, "mean_b": mb, "median_a": mda, "median_b": mdb,
            "mean_diff": ma - mb, "median_diff": mda - mdb,
            "ratio_mean": (ma / mb) if (mb and abs(mb) > 1e-12) else float("nan"),
            "cohens_d": cohens_d(b, a), "ks": ks_2samp(a, b)}


# --------------------------------------------------------------------------- correlations (§16/§17)
def correlate(x, y) -> dict:
    """Pearson + Spearman of two per-scene vectors (finite-paired)."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return {"n": int(m.sum()), "pearson": float("nan"), "spearman": float("nan")}
    return {"n": int(m.sum()), "pearson": pearson(x[m], y[m]), "spearman": spearman(x[m], y[m])}
