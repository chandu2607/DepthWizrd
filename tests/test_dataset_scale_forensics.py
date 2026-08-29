#!/usr/bin/env python
"""Standalone tests for the DATASET / SCALE FORENSICS utilities (Experiment #7).

Run: python tests/test_dataset_scale_forensics.py   (no pytest needed)

Covers `depthwizard.diagnostics.scale_forensics` -- the pure-numeric Part-B toolkit:
raw raster metadata (never fabricating a GSD, §10), height/depth distribution stats,
histogram overlap + KS, tall-band depth stats, connected-component footprint / border
(incl. the numpy flood-fill fallback), split-shift quantification and correlation.

Each test recomputes the expected quantity from the same inputs with an independent
numpy expression wherever possible, so the test verifies the function AGREES with the
definition rather than echoing a hard-coded constant. NOTHING here trains or touches
the model / DA-V2 / cache; the module is side-effect-free by design.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from depthwizard.diagnostics import scale_forensics as sf  # noqa: E402


# ---------------------------------------------------------------- helpers
def test_finite_strips_nonfinite():
    x = np.array([1.0, np.nan, np.inf, -np.inf, 2.0])
    assert np.array_equal(sf.finite(x), np.array([1.0, 2.0]))
    assert sf.finite(np.array([np.nan, np.inf])).size == 0


def test_tk_formatting():
    assert sf._tk(5.0) == "5" and sf._tk(20.0) == "20"
    assert sf._tk(2.5) == "2.5"


def test_basic_matches_numpy():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    b = sf._basic(x)
    assert b["n"] == 4
    assert abs(b["mean"] - 2.5) < 1e-12 and abs(b["median"] - 2.5) < 1e-12
    assert b["min"] == 1.0 and b["max"] == 4.0
    assert "p50" in b and abs(b["p50"] - 2.5) < 1e-12
    e = sf._basic(np.array([np.nan, np.nan]))
    assert e["n"] == 0 and np.isnan(e["mean"]) and np.isnan(e["p50"])


def test_frac_above_matches_definition():
    x = np.array([1.0, 6.0, 11.0, 16.0, 21.0, 31.0, 41.0])
    fa = sf.frac_above(x, thresholds=(5.0, 10.0, 15.0, 20.0, 30.0, 40.0))
    for t in (5.0, 10.0, 15.0, 20.0, 30.0, 40.0):
        assert abs(fa[f">{sf._tk(t)}m"] - float(np.mean(x > t))) < 1e-12
    assert all(np.isnan(v) for v in sf.frac_above(np.array([])).values())


# ---------------------------------------------------------------- raw raster metadata (§10/§11)
def test_raw_raster_info_tif_never_fabricates_gsd():
    import tifffile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "OMA_0001_AGL.tif")
        arr = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
        tifffile.imwrite(p, arr)
        info = sf.raw_raster_info(p)
        assert info["exists"] and info["format"] == "tif"
        assert tuple(info["shape"]) == (16, 16)
        assert info["dtype"] == "float32"
        # a plain (non-geo) TIFF must NOT be reported as georeferenced and must NOT invent a GSD
        assert info["georeferenced"] is False
        assert info["gsd_m"] is None


def test_raw_raster_info_png_shape_and_dtype():
    from PIL import Image
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "JAX_0001_RGB.png")
        arr = np.zeros((12, 20, 3), np.uint8)
        Image.fromarray(arr).save(p)
        info = sf.raw_raster_info(p)
        assert info["exists"] and info["format"] == "png"
        # shape is (H, W, C); PIL size is (W, H) so this also checks the axis order
        assert tuple(info["shape"]) == (12, 20, 3)
        assert info["dtype"] == "uint8"
        assert info["gsd_m"] is None


def test_raw_raster_info_missing_file():
    info = sf.raw_raster_info("does_not_exist_12345.tif")
    assert info["exists"] is False and info["shape"] is None


def test_summarize_raw_infos_counts_and_gsd_flag():
    import tifffile
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i in range(2):
            p = os.path.join(td, f"t{i}.tif")
            tifffile.imwrite(p, np.zeros((8, 8), np.float32))
            paths.append(p)
        infos = [sf.raw_raster_info(p) for p in paths]
        infos.append(sf.raw_raster_info(os.path.join(td, "missing.tif")))
        summ = sf.summarize_raw_infos(infos)
        assert summ["n"] == 3 and summ["n_readable"] == 2
        assert summ["shapes"].get("(8, 8)") == 2
        assert summ["gsd_available"] is False


# ---------------------------------------------------------------- height distribution (§12/§18)
def test_height_stats_matches_masked_numpy():
    d = np.ones((4, 4), np.float64)
    h = np.array([[-999., 10., 20., 2.],
                  [0.,     6.,  6., 3.],
                  [6.,     6., 30., 2.],
                  [2.,     2.,  2., 2.]], np.float64)
    cls = np.array([[2, 6, 6, 2],
                    [2, 6, 6, 2],
                    [6, 6, 6, 2],
                    [2, 2, 2, 2]], np.int32)
    out = sf.height_stats(h, d, cls=cls, blabel=6, nodata=-999.0)
    vm = np.isfinite(h) & (h != -999.0) & np.isfinite(d)
    assert out["n_valid"] == int(vm.sum())
    assert abs(out["ground_frac"] - float(np.mean(cls[vm] == 2))) < 1e-12
    assert abs(out["building_frac"] - float(np.mean(cls[vm] == 6))) < 1e-12
    assert out["n_building"] == int((vm & (cls == 6)).sum())
    assert abs(out["all"]["frac_above"][">5m"] - float(np.mean(h[vm] > 5))) < 1e-12
    hb = h[vm & (cls == 6)]
    assert abs(out["building"]["median"] - float(np.median(hb))) < 1e-12


def test_height_stats_without_cls():
    d = np.ones((3, 3), np.float64)
    h = np.full((3, 3), 8.0)
    out = sf.height_stats(h, d, cls=None, nodata=-999.0)
    assert np.isnan(out["ground_frac"]) and np.isnan(out["building_frac"])
    assert out["n_building"] == 0 and out["building"]["n"] == 0
    assert out["all"]["n"] == 9


# ---------------------------------------------------------------- depth distribution (§13/§19)
def test_depth_stats_histogram_conserves_count():
    d = np.arange(100, dtype=np.float64)
    out = sf.depth_stats(d, hist_bins=40)
    assert out["n"] == 100
    assert len(out["hist_edges"]) == len(out["hist_counts"]) + 1
    assert sum(out["hist_counts"]) == 100          # all finite pixels land in a bin
    assert abs(out["median"] - float(np.median(d))) < 1e-9


def test_hist_overlap_bounds():
    assert abs(sf.hist_overlap([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-12   # identical -> 1
    assert sf.hist_overlap([1, 0, 0], [0, 0, 1]) == 0.0               # disjoint -> 0
    assert np.isnan(sf.hist_overlap([1, 2, 3], [1, 2]))               # length mismatch -> nan


def test_ks_2samp_extremes():
    x = np.arange(100, dtype=np.float64)
    assert sf.ks_2samp(x, x.copy()) == 0.0                            # identical -> 0
    assert sf.ks_2samp(x, x + 1000.0) > 0.95                          # fully separated -> ~1
    assert np.isnan(sf.ks_2samp(np.array([]), x))


def test_tall_band_depth_stats_bins_by_height():
    # one building pixel per tall band, each with a known depth; ground elsewhere
    h = np.array([[17., 25.], [35., 45.]], np.float64)
    d = np.array([[1.0, 2.0], [3.0, 4.0]], np.float64)
    cls = np.full((2, 2), 6, np.int32)
    out = sf.tall_band_depth_stats(d, h, cls=cls, blabel=6, nodata=-999.0)
    assert out["15_20"]["n"] == 1 and abs(out["15_20"]["depth_median"] - 1.0) < 1e-12
    assert out["20_30"]["n"] == 1 and abs(out["20_30"]["depth_median"] - 2.0) < 1e-12
    assert out["30_40"]["n"] == 1 and abs(out["30_40"]["depth_median"] - 3.0) < 1e-12
    assert out["40_inf"]["n"] == 1 and abs(out["40_inf"]["depth_median"] - 4.0) < 1e-12


def test_tall_band_empty_is_nan():
    h = np.full((3, 3), 2.0)      # nothing tall
    d = np.ones((3, 3))
    out = sf.tall_band_depth_stats(d, h, cls=np.full((3, 3), 6, np.int32))
    assert out["15_20"]["n"] == 0 and np.isnan(out["15_20"]["depth_median"])


# ---------------------------------------------------------------- connected components (§20/§21)
def _two_blob_mask():
    m = np.zeros((5, 5), bool)
    m[0:2, 0:2] = True   # blob of 4 (touches top-left edge)
    m[4, 4] = True        # blob of 1 (touches bottom-right edge)
    return m


def test_label_components_counts_blobs():
    lab, n = sf.label_components(_two_blob_mask())
    assert n == 2
    sizes = sorted(np.bincount(lab.ravel())[1:].tolist())
    assert sizes == [1, 4]


def test_flood_label_fallback_matches():
    # exercise the numpy fallback directly (independent of scipy availability)
    lab, n = sf._flood_label(_two_blob_mask())
    assert n == 2
    assert sorted(np.bincount(lab.ravel())[1:].tolist()) == [1, 4]


def test_footprint_stats_sizes():
    fp = sf.footprint_stats(_two_blob_mask(), large_px=3)
    assert fp["building_px"] == 5 and fp["n_components"] == 2
    assert fp["largest_px"] == 4 and fp["n_large"] == 1
    empty = sf.footprint_stats(np.zeros((4, 4), bool))
    assert empty["n_components"] == 0 and empty["largest_px"] == 0


def test_border_stats_edge_touching():
    bs = sf.border_stats(_two_blob_mask(), border=1)
    assert bs["n_components"] == 2 and bs["n_border_components"] == 2
    assert abs(bs["border_component_frac"] - 1.0) < 1e-12
    assert 0.0 < bs["border_px_frac"] <= 1.0
    # a fully interior single pixel touches no edge
    interior = np.zeros((7, 7), bool); interior[3, 3] = True
    bi = sf.border_stats(interior, border=1)
    assert bi["n_border_components"] == 0 and bi["border_px_frac"] == 0.0


# ---------------------------------------------------------------- split shift + correlation (§15/§16/§17/§23)
def test_split_shift_known_means():
    a = np.array([1.0, 2.0, 3.0, 4.0])   # mean 2.5
    b = np.array([2.0, 4.0, 6.0, 8.0])   # mean 5.0
    sh = sf.split_shift(a, b)
    assert abs(sh["mean_a"] - 2.5) < 1e-12 and abs(sh["mean_b"] - 5.0) < 1e-12
    assert abs(sh["mean_diff"] - (-2.5)) < 1e-12
    assert abs(sh["ratio_mean"] - 0.5) < 1e-12
    assert abs(sh["median_a"] - 2.5) < 1e-12 and abs(sh["median_b"] - 5.0) < 1e-12
    assert np.isfinite(sh["cohens_d"]) and sh["cohens_d"] < 0        # a < b
    assert 0.0 <= sh["ks"] <= 1.0


def test_correlate_monotonic_and_small_n():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 2.0 * x
    c = sf.correlate(x, y)
    assert c["n"] == 5 and abs(c["pearson"] - 1.0) < 1e-9 and abs(c["spearman"] - 1.0) < 1e-9
    small = sf.correlate(np.array([1.0, 2.0]), np.array([2.0, 4.0]))
    assert small["n"] == 2 and np.isnan(small["pearson"]) and np.isnan(small["spearman"])


def test_module_does_not_import_torch():
    import subprocess
    code = ("import sys; import depthwizard.diagnostics.scale_forensics as m; "
            "assert 'torch' not in sys.modules, 'scale_forensics must not import torch'; "
            "print('ok')")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, f"pure-module torch check failed: {r.stderr}\n{r.stdout}"


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
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
