"""Bounded-effort dataset acquisition for Phase-1.

Policy (from the experiment spec): spend a bounded amount of wall-clock trying to
obtain real DFC2019 data. If it does not resolve quickly, fall back to a clearly
labelled SYNTHETIC smoke-test set so the pipeline can still be exercised -- but
synthetic numbers are NOT feasibility evidence and the report marks them so.

Resolution order:
  1. cfg.data.root already contains RGB/AGL triplets  -> use as-is (offline, best).
  2. source == 'hf_mirror'  -> huggingface_hub.snapshot_download, then scan triplets.
  3. source == 'hf_blocks'  -> block-tiled mirror ({split}/{rgb,depth,seg}); adapter
                               in data/hf_blocks.py decodes color seg PNGs to CLS.
  4. source == 'ieee'       -> cannot be automated (login/EULA); print instructions.
  5. source == 'synthetic'  -> generate offline smoke-test tiles.

The HF mirror schema is inspected at RUNTIME (we scan whatever files land on disk
for *_RGB.tif / *_AGL.tif pairs) rather than assuming a layout we could not verify.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from .datasets import Record, scan_dfc_dir, cities_present, make_synthetic_city


IEEE_INSTRUCTIONS = """\
Real DFC2019 Track-1 (US3D) requires a free IEEE DataPort login + EULA, so it
cannot be fetched non-interactively. Options:
  * Kaggle:  search 'DFC2019' / 'US3D' datasets and Add to a Kaggle notebook.
  * IEEE DataPort: https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019
  * HF mirror: set data.source: hf_mirror (schema inspected at runtime).
Place the extracted *_RGB.tif / *_AGL.tif / *_CLS.tif under data.root and re-run.
"""


def _has_triplets(root: str, cfg_data) -> bool:
    if not root or not Path(root).exists():
        return False
    recs = scan_dfc_dir(root, cfg_data.rgb_suffix, cfg_data.agl_suffix, cfg_data.cls_suffix)
    return len(recs) > 0


def try_hf_mirror(repo: str, out_dir: str, budget_s: float) -> str | None:
    """Download an HF dataset snapshot within a time budget. Returns dir or None."""
    t0 = time.time()
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print(f"[fetch] huggingface_hub unavailable ({e}); skipping HF mirror.")
        return None
    try:
        print(f"[fetch] snapshot_download('{repo}') -> {out_dir} (budget {budget_s:.0f}s)")
        path = snapshot_download(repo_id=repo, repo_type="dataset", local_dir=out_dir)
        print(f"[fetch] HF snapshot done in {time.time()-t0:.0f}s at {path}")
        return path
    except Exception as e:
        print(f"[fetch] HF mirror failed after {time.time()-t0:.0f}s: {e}")
        return None


def resolve_records(cfg) -> tuple[str, list[Record]]:
    """Return (source_tag, records). Real data preferred; synthetic returns []."""
    d = cfg.data
    # 1. already-present triplets win regardless of configured source
    if _has_triplets(d.root, d):
        recs = scan_dfc_dir(d.root, d.rgb_suffix, d.agl_suffix, d.cls_suffix)
        print(f"[fetch] found {len(recs)} triplets in {d.root}: {cities_present(recs)}")
        return "local", recs

    if d.source == "hf_mirror":
        out = try_hf_mirror(d.hf_repo, os.path.join(d.root, "_hf"), budget_s=1800)
        if out and _has_triplets(out, d):
            recs = scan_dfc_dir(out, d.rgb_suffix, d.agl_suffix, d.cls_suffix)
            print(f"[fetch] HF mirror gave {len(recs)} triplets: {cities_present(recs)}")
            return "hf_mirror", recs
        print("[fetch] HF mirror produced no recognizable triplets.")

    if d.source == "hf_blocks":
        # Block-tiled mirror ({split}/{rgb,depth,seg}); see data/hf_blocks.py.
        # Real DFC crops but a non-canonical layout, so it needs a dedicated
        # adapter that decodes the color seg PNGs into integer CLS masks.
        from . import hf_blocks
        snap = hf_blocks.ensure_local(d.hf_repo)
        if snap:
            cls_cache = os.path.join(d.root, "_hf_cls")
            cities = (set(cfg.split.train_cities) | set(cfg.split.val_cities)
                      | set(cfg.split.test_cities))
            recs = hf_blocks.scan_hf_blocks(snap, cls_cache, cities=cities)
            if recs:
                print(f"[fetch] hf_blocks gave {len(recs)} records "
                      f"(unofficial mirror, preprocessed nDSM): {cities_present(recs)}")
                return "hf_blocks", recs
        print("[fetch] hf_blocks produced no records.")

    if d.source == "ieee":
        print(IEEE_INSTRUCTIONS)

    return "synthetic", []


def synthetic_samples(cfg) -> tuple[list[dict], list[dict], list[dict]]:
    """Two synthetic 'cities' with different height stats: train on one, test on
    the other, mirroring the city-held-out protocol. SMOKE TEST ONLY."""
    ts = cfg.data.tile_size
    n = min(cfg.data.max_tiles_per_city or 24, 24)
    train_city = make_synthetic_city(n, "SYNA", ts, seed=1, max_building_h=40.0)
    test_city = make_synthetic_city(max(n // 2, 8), "SYNB", ts, seed=2, max_building_h=70.0)
    k = max(int(len(train_city) * cfg.split.val_fraction_within_train_city), 1)
    val = train_city[:k]
    train = train_city[k:]
    return train, val, test_city
