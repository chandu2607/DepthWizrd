"""Dataset loading for DepthWizard Phase-1.

Primary target: DFC2019 Track-1 (US3D) canonical layout -- triplets of
    <tile>_RGB.tif  (uint8 RGB, ~0.35 m GSD)
    <tile>_AGL.tif  (float32 above-ground height in METERS = nDSM; nodata sentinel)
    <tile>_CLS.tif  (uint8 semantic labels; 2 ground,5 tree,6 building,9 water,17 bridge)
City is the filename prefix (e.g. JAX_*, OMA_*), which is what enables the
MANDATORY city-held-out generalization test -- never random-split same-city tiles.

Also provides:
- a robust multi-backend raster reader (rasterio -> tifffile -> PIL),
- a SYNTHETIC generator for smoke-testing the full pipeline WITHOUT any download
  (its numbers are NOT valid feasibility evidence and are labelled as such).
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Raster IO
# --------------------------------------------------------------------------- #
def read_raster(path: str) -> np.ndarray:
    """Read a raster to a numpy array (HxW or HxWxC). Tries rasterio, tifffile, PIL."""
    try:
        import rasterio
        with rasterio.open(path) as ds:
            arr = ds.read()  # CxHxW
            arr = np.transpose(arr, (1, 2, 0)) if arr.shape[0] > 1 else arr[0]
            return arr
    except Exception:
        pass
    try:
        import tifffile
        return tifffile.imread(path)
    except Exception:
        pass
    from PIL import Image
    return np.array(Image.open(path))


# --------------------------------------------------------------------------- #
# DFC2019 canonical layout
# --------------------------------------------------------------------------- #
@dataclass
class Record:
    tile_id: str
    city: str
    rgb_path: str
    agl_path: str
    cls_path: Optional[str]


_CITY_RE = re.compile(r"^([A-Za-z]+)")


def scan_dfc_dir(root: str, rgb_suffix="_RGB.tif", agl_suffix="_AGL.tif",
                 cls_suffix="_CLS.tif") -> list[Record]:
    """Find all RGB/AGL(/CLS) triplets under `root` (recursively)."""
    root = str(root)
    records: list[Record] = []
    rgb_files = glob.glob(os.path.join(root, "**", f"*{rgb_suffix}"), recursive=True)
    for rgb in sorted(rgb_files):
        base = rgb[: -len(rgb_suffix)]
        agl = base + agl_suffix
        cls = base + cls_suffix
        if not os.path.exists(agl):
            continue
        tile = os.path.basename(base)
        m = _CITY_RE.match(tile)
        city = m.group(1).upper() if m else "UNK"
        records.append(Record(tile, city, rgb, agl,
                              cls if os.path.exists(cls) else None))
    return records


def cities_present(records: list[Record]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        out[r.city] = out.get(r.city, 0) + 1
    return out


def load_sample(rec: Record, tile_size: int, nodata: float,
                depth_model=None) -> dict:
    """Load one Record into a Sample dict (rgb, gt, cls, depth, city, id)."""
    import cv2

    rgb = read_raster(rec.rgb_path)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        # scale 11/16-bit imagery to 8-bit via robust percentiles
        lo, hi = np.percentile(rgb, [2, 98])
        rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)

    agl = read_raster(rec.agl_path).astype(np.float32)
    if agl.ndim == 3:
        agl = agl[..., 0]
    agl = np.where(agl == nodata, np.nan, agl)

    cls = None
    if rec.cls_path:
        cls = read_raster(rec.cls_path)
        if cls.ndim == 3:
            cls = cls[..., 0]

    # resize everything to a common tile size (RGB/AGL bilinear/nearest, CLS nearest)
    rgb = cv2.resize(rgb, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
    agl = cv2.resize(agl, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)
    if cls is not None:
        cls = cv2.resize(cls.astype(np.int32), (tile_size, tile_size),
                         interpolation=cv2.INTER_NEAREST)

    sample = {"rgb": rgb, "gt": agl, "cls": cls, "city": rec.city, "id": rec.tile_id}
    if depth_model is not None:
        sample["depth"] = depth_model.infer(rgb, key=rec.tile_id,
                                             target_hw=(tile_size, tile_size))
    return sample


# --------------------------------------------------------------------------- #
# City-held-out splitting (MANDATORY generalization protocol)
# --------------------------------------------------------------------------- #
def split_by_city(records: list[Record], train_cities, val_cities, test_cities,
                  val_fraction_within_train_city: float, seed: int,
                  max_tiles_per_city: int = 0):
    rng = np.random.default_rng(seed)
    by_city: dict[str, list[Record]] = {}
    for r in records:
        by_city.setdefault(r.city, []).append(r)
    for c in by_city:
        rng.shuffle(by_city[c])
        if max_tiles_per_city > 0:
            by_city[c] = by_city[c][:max_tiles_per_city]

    train, val, test = [], [], []
    for c in train_cities:
        recs = by_city.get(c, [])
        n_val = int(len(recs) * val_fraction_within_train_city)
        val += recs[:n_val]
        train += recs[n_val:]
    # explicit val cities (if different) add held-out same-domain tiles
    for c in val_cities:
        if c not in train_cities:
            val += by_city.get(c, [])
    for c in test_cities:
        test += by_city.get(c, [])
    return train, val, test


# --------------------------------------------------------------------------- #
# Synthetic smoke-test data (NOT valid feasibility evidence)
# --------------------------------------------------------------------------- #
def make_synthetic_city(n: int, city: str, tile_size: int, seed: int,
                        max_building_h: float) -> list[dict]:
    """Procedural RGB + nDSM + CLS. Two calls with different max_building_h give
    two 'cities' with different height distributions, so the harness can be
    exercised end-to-end offline. Results are for PLUMBING ONLY."""
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n):
        H = W = tile_size
        # low-frequency terrain (we predict nDSM so terrain is only for the RGB look)
        base = rng.normal(0, 1, (8, 8)).astype(np.float32)
        import cv2
        terrain = cv2.resize(base, (W, H), interpolation=cv2.INTER_CUBIC)
        ndsm = np.zeros((H, W), np.float32)
        cls = np.full((H, W), 2, np.int32)  # ground
        # random buildings
        for _ in range(rng.integers(3, 12)):
            bw, bh = rng.integers(20, 70, size=2)
            x, y = rng.integers(0, W - bw), rng.integers(0, H - bh)
            h = float(rng.uniform(3, max_building_h))
            ndsm[y:y + bh, x:x + bw] = h
            cls[y:y + bh, x:x + bw] = 6  # building
        # trees
        for _ in range(rng.integers(5, 20)):
            r = rng.integers(4, 12)
            x, y = rng.integers(r, W - r), rng.integers(r, H - r)
            yy, xx = np.ogrid[:H, :W]
            mask = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
            ndsm[mask] = float(rng.uniform(2, 10))
            cls[mask] = 5
        # fake RGB: shade by (terrain + ndsm) so there is *some* structure signal
        shade = terrain + ndsm / max_building_h
        shade = (shade - shade.min()) / (np.ptp(shade) + 1e-6)
        rgb = (np.stack([shade, shade * 0.8 + 0.1, 1 - shade], -1) * 255).astype(np.uint8)
        samples.append({"rgb": rgb, "gt": ndsm, "cls": cls,
                        "city": city, "id": f"{city}_{i:04d}"})
    return samples
