"""Adapter for the `JasonXF/DFC2019-10k` HuggingFace mirror (block-tiled layout).

This mirror is NOT the canonical DFC2019 triplet layout. Per split it stores
three parallel modality folders:

    {train,val,test}/rgb/<NAME>.tif      uint8 RGB orthophoto crop
    {train,val,test}/depth/<NAME>.tif    float32 nDSM in METERS (ground floored to 0)
    {train,val,test}/seg/<NAME>.png      3-channel COLOR semantic map (NOT class codes)

where <NAME> encodes the city, e.g. ``JAX_004_015_0_0`` / ``OMA_026_029_0_512``.

PROVENANCE (verified 2026-08-26): the rgb/depth/seg crops are REAL DFC2018/19
orthophotos + LiDAR-derived nDSM (confirmed by visual inspection: coherent
aerial detail, no generative artifacts, leafless-winter scenes that contradict
the repo's "lush green" image-gen prompt.json; that prompt.json / the parent
"SynthUrbanSAT" name belong to a separate synthetic branch, not these files).
The depth is a cleaned/derived nDSM product: ground == 0.0, no negative values,
no ``-999`` nodata sentinel. It is USABLE feasibility evidence, but it is an
UNOFFICIAL mirror -- the official IEEE GRSS DFC2019 (login + EULA) remains the
authoritative upgrade path for final reported numbers.

We map this into the repo's canonical ``Record`` / ``load_sample`` path with
ZERO changes to the loader by:
  * pointing ``rgb_path`` -> ``rgb/<NAME>.tif`` and ``agl_path`` -> ``depth/<NAME>.tif``
    directly (both already match what ``load_sample`` expects), and
  * DECODING the color seg PNG to integer DFC class codes and caching a small
    single-channel ``<NAME>_CLS.tif`` that ``load_sample`` reads as usual.

The seg color->class palette was decoded EMPIRICALLY (nDSM-height correlation +
a rooftop overlay check), never assumed from documentation:

    (255,  0,  0)  red    -> 6 building   (CONFIRMED: red pixels sit on rooftops)
    (  0,255,  0)  green  -> 5 tree        (tall, textured canopies)
    (  0,225,255)  cyan   -> 9 water       (CONFIRMED: the pond)
    (  0,  0,255)  blue   -> 2 ground      (street network; ground level)
    (128,  0,128)  purple -> 2 ground      (lawns / parcels; ground level)
    (  0,  0,  0)  black  -> 0 unlabeled

Only ``red -> 6`` affects the building / non-building metric split
(``compute_class_metrics`` keys solely on ``== building_label``); the other
codes are best-effort and do not change any reported building number.

The mirror's own ``train/val/test`` dirs are IGNORED for splitting -- we pool
every tile and re-split by CITY downstream (train JAX / in-domain val JAX /
test OMA) to enforce the mandatory city-held-out generalization protocol.
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import numpy as np

from .datasets import Record

_CITY_RE = re.compile(r"^([A-Za-z]+)")

# Empirically decoded palette (see module docstring). Colors are (R, G, B).
SEG_PALETTE = {
    (255, 0, 0): 6,      # building  (VISUALLY CONFIRMED on rooftops)
    (0, 255, 0): 5,      # tree
    (0, 225, 255): 9,    # water
    (0, 0, 255): 2,      # ground (street grid)
    (128, 0, 128): 2,    # ground (lawns / parcels)
    (0, 0, 0): 0,        # unlabeled
}


def decode_seg_to_cls(seg: np.ndarray) -> tuple[np.ndarray, float]:
    """Map an HxWx3 color seg map to uint8 DFC class codes.

    Returns ``(cls, unmatched_fraction)``. Colors not in ``SEG_PALETTE`` map to
    0 (unlabeled). ``cls`` is uint8 (all class codes are small), which the
    canonical ``load_sample`` re-reads and up-casts before its nearest-neighbour
    resize.
    """
    if seg.ndim == 2:
        seg = np.stack([seg] * 3, axis=-1)
    seg = seg[..., :3].astype(np.int64)
    r, g, b = seg[..., 0], seg[..., 1], seg[..., 2]
    cls = np.zeros(seg.shape[:2], dtype=np.uint8)
    matched = np.zeros(seg.shape[:2], dtype=bool)
    for (cr, cg, cb), code in SEG_PALETTE.items():
        m = (r == cr) & (g == cg) & (b == cb)
        cls[m] = code
        matched |= m
    return cls, float(1.0 - matched.mean())


def ensure_local(repo: str, allow_patterns=("train/**", "val/**"),
                 max_workers: int = 4, attempts: int = 8) -> str | None:
    """Download the mirror snapshot (to the HF cache, NOT a local OneDrive copy).

    By default fetches only ``train/`` + ``val/`` -- that already contains every
    JAX tile (train) and every OMA tile (train + val); the ``test/`` split is the
    anonymized ``block_*`` set we never use. Returns the snapshot dir or None.

    ``snapshot_download`` is resumable (already-cached files are skipped by etag),
    so on a transient failure -- notably HF's HTTP 429 rate limit, which this
    2997-file dataset trips near the end with the default 8 workers -- we retry
    with backoff and each attempt only fetches what is still missing. ``max_workers``
    is kept modest to stay under the rate limit.
    """
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:  # pragma: no cover - env-dependent
        print(f"[hf_blocks] huggingface_hub unavailable: {e}")
        return None
    import time
    for i in range(1, attempts + 1):
        try:
            return snapshot_download(
                repo_id=repo, repo_type="dataset",
                allow_patterns=list(allow_patterns), max_workers=max_workers)
        except Exception as e:
            msg = str(e)
            rate_limited = ("429" in msg) or ("Too Many Requests" in msg)
            print(f"[hf_blocks] snapshot attempt {i}/{attempts} failed"
                  f"{' (rate-limited)' if rate_limited else ''}: {msg[:180]}")
            if i == attempts:
                print("[hf_blocks] giving up after retries.")
                return None
            wait = 30 * i if rate_limited else 10
            print(f"[hf_blocks] retrying in {wait}s (resumable; cached files skipped)...")
            time.sleep(wait)
    return None


def scan_hf_blocks(root: str, cls_cache_dir: str, cities=None) -> list[Record]:
    """Build ``Record``s from the block-tiled mirror rooted at ``root``.

    Pairs ``rgb`` / ``depth`` / ``seg`` by basename across all split dirs found
    under ``root``. Decodes each seg PNG to an integer CLS ``.tif`` cached under
    ``cls_cache_dir`` (skipped if already cached). ``tile_id`` is prefixed with
    the split-dir name so it is globally unique -- essential because the depth
    prior is disk-cached by ``tile_id`` and OMA appears in both ``train`` and
    ``val``. City (for the held-out split) is taken from the basename, never the
    prefixed id. Pass ``cities`` (a set) to skip tiles of other cities up front
    and avoid decoding their seg maps.
    """
    import tifffile
    from PIL import Image

    root = str(root)
    Path(cls_cache_dir).mkdir(parents=True, exist_ok=True)
    records: list[Record] = []
    rgb_files = sorted(glob.glob(os.path.join(root, "**", "rgb", "*.tif"), recursive=True))
    warned = 0
    for rgb in rgb_files:
        name = Path(rgb).stem
        split_dir = str(Path(rgb).parents[1])
        split_tag = Path(split_dir).name
        depth = os.path.join(split_dir, "depth", name + ".tif")
        if not os.path.exists(depth):
            continue
        m = _CITY_RE.match(name)
        city = m.group(1).upper() if m else "UNK"
        if cities is not None and city not in cities:
            continue
        seg = os.path.join(split_dir, "seg", name + ".png")
        cls_path = None
        if os.path.exists(seg):
            cp = os.path.join(cls_cache_dir, f"{split_tag}__{name}_CLS.tif")
            if not os.path.exists(cp):
                arr = np.array(Image.open(seg).convert("RGB"))
                cls, unmatched = decode_seg_to_cls(arr)
                if unmatched > 0.02 and warned < 5:
                    print(f"[hf_blocks] {name}: {unmatched * 100:.1f}% seg pixels "
                          f"unmatched -> 0 (unlabeled)")
                    warned += 1
                tifffile.imwrite(cp, cls)
            cls_path = cp
        tile_id = f"{split_tag}__{name}"
        records.append(Record(tile_id, city, rgb, depth, cls_path))
    return records
