"""Configuration handling for DepthWizard experiments.

Uses a small dataclass tree loaded from YAML so every run is reproducible and
paths are configurable (no hard-coded absolute paths). Unknown keys in the
YAML are ignored with a warning rather than crashing, so the config file can
carry documentation keys.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except Exception as e:  # pragma: no cover
    yaml = None
    _YAML_ERR = e


@dataclass
class DataConfig:
    # Root dir holding the dataset (canonical DFC2019 layout OR a fetched mirror).
    root: str = "data/dfc2019"
    # Source selector, see depthwizard/data/fetch.py: {ieee, hf_mirror, synthetic, byo}
    source: str = "synthetic"
    hf_repo: str = "JasonXF/DFC2019-10k"
    # File suffixes for the canonical DFC2019 Track-1 layout.
    rgb_suffix: str = "_RGB.tif"
    agl_suffix: str = "_AGL.tif"   # AGL = above-ground-level = nDSM (meters)
    cls_suffix: str = "_CLS.tif"
    nodata: float = -999.0
    building_label: int = 6
    tile_size: int = 512          # tiles are resized to this for CPU/GPU speed
    max_tiles_per_city: int = 200  # cap for bounded runtime; 0 = no cap


@dataclass
class DepthConfig:
    model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    input_size: int = 518
    cache_dir: str = "data/depth_cache"
    use_cache: bool = True


@dataclass
class SplitConfig:
    # City-held-out generalization (MANDATORY). Do NOT random-split same-city tiles.
    train_cities: list[str] = field(default_factory=lambda: ["JAX"])
    val_cities: list[str] = field(default_factory=lambda: ["JAX"])   # held-out tiles, same city
    test_cities: list[str] = field(default_factory=lambda: ["OMA"])  # fully unseen city
    val_fraction_within_train_city: float = 0.15
    seed: int = 1337


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 4
    lr: float = 1e-3
    width: int = 24          # base channel width of the small fusion head
    train_res: int = 256     # train the head at reduced resolution for speed
    # Phase-5 reconstruction-fidelity variable: the fusion-head architecture.
    #   "unet3" -> original 3-level SmallFusionUNet (DEFAULT: C_none / C_log1p stay
    #              bit-for-bit reproducible; the whole Phase-1..4 record is unchanged).
    #   "unet4" -> SmallReconUNet, ONE extra pooling/decoder level (bottleneck 32->16
    #              at train_res=256), ~doubling the effective receptive field so large
    #              structures can be reconstructed across their FULL footprint. The new
    #              deepest level is held at w*4 channels (NOT doubled) -> the added
    #              params buy receptive field/depth, not raw width. Single-variable
    #              vs C_log1p (everything else identical).
    arch: str = "unet3"
    max_train_pixels_affine: int = 2_000_000  # subsample for affine lstsq
    amp: bool = True         # mixed precision if CUDA available
    num_workers: int = 2
    # Phase-2 target-space transform for the learned head (Baseline C only).
    #   "none"  -> train on raw metric nDSM (the ORIGINAL Baseline C; keep default
    #              so Phase-1 C remains bit-for-bit reproducible, per the spec).
    #   "log1p" -> train on log(1+h), invert with expm1 at predict BEFORE resizing.
    #              Justified only by the target-distribution evidence (heavy right
    #              skew + ground dominance); see scripts/phase2_diagnose_distribution.py.
    target_transform: str = "none"
    # Phase-3 height-aware loss weighting (Baseline C only).
    #   "standard"        -> plain masked-L1 (keep default so C_none / C_log1p stay
    #                        bit-for-bit reproducible; the standard path is untouched).
    #   "height_weighted" -> per-pixel weighted masked-L1 with weight w(h) derived
    #                        from PHYSICAL height h (meters, pre-transform):
    #                          w(h) = min(1 + max(h,0)/loss_weight_scale, loss_weight_max)
    #                        Weighted MEAN (÷Σw) so loss magnitude ~ unweighted -> the
    #                        effective LR is unchanged; the ONLY variable vs C_log1p is
    #                        per-pixel emphasis. Scale/cap are TRAINING-derived (see
    #                        scripts/phase3_weight_diagnostic.py); no test info leaks.
    #   "tail_weighted"   -> Phase-4 CALIBRATED tail weight. Same masked-L1 machinery,
    #                        but FLAT (w=1) through the abundant low/moderate regime and
    #                        rising ONLY past a training-derived threshold, with a
    #                        gentler cap:
    #                          w(h) = min(1 + max(h - loss_tail_start,0)/loss_tail_scale,
    #                                     loss_tail_max)   on PHYSICAL height h.
    #                        Motivation (Phase-3): the height_weighted ramp began at h=0
    #                        and shifted the WHOLE distribution up, damaging the 0–15 m
    #                        population (91.7% of JAX-train px). The threshold protects
    #                        that regime; all three params are JAX-train-derived (see
    #                        scripts/phase4_weight_diagnostic.py); no test info leaks.
    loss_type: str = "standard"
    loss_weight_scale: float = 7.0   # h_scale (m) ~ JAX-train building-pixel median (7.16)
    loss_weight_max: float = 5.0     # w_max cap: bounds weights -> stable, no tall-outlier blowup
    # Phase-4 tail_weighted params (used only when loss_type="tail_weighted").
    loss_tail_start: float = 15.0    # w=1 for h<=this: onset of the sparse tail (measured ~P92 all / ~P79 bldg ≈ the ~14 m learned ceiling)
    loss_tail_scale: float = 12.5    # ramp scale above the threshold (reaches the cap at h_start+(cap-1)*scale = 40 m ≈ P99.3)
    loss_tail_max: float = 3.0       # gentler cap than height_weighted (5.0); rare extremes cannot dominate the gradient


@dataclass
class ExperimentConfig:
    out_dir: str = "runs/phase1"
    baselines: list[str] = field(default_factory=lambda: ["A", "B", "C"])
    seeds: list[int] = field(default_factory=lambda: [0, 1])  # reproducibility check
    data: DataConfig = field(default_factory=DataConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _build(dc_type, d: dict):
    """Instantiate a dataclass from a dict, ignoring unknown keys."""
    if not isinstance(d, dict):
        return dc_type()
    known = {f.name: f for f in fields(dc_type)}
    kwargs = {}
    for k, v in d.items():
        if k not in known:
            warnings.warn(f"[config] ignoring unknown key '{k}' for {dc_type.__name__}")
            continue
        ftype = known[k].type
        # Recurse into nested dataclasses.
        if hasattr(ftype, "__dataclass_fields__") and isinstance(v, dict):
            kwargs[k] = _build(ftype, v)
        else:
            kwargs[k] = v
    return dc_type(**kwargs)


def load_config(path: Optional[str] = None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig()
    if yaml is None:  # pragma: no cover
        raise RuntimeError(f"pyyaml required to load configs: {_YAML_ERR}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    # Nested dataclasses need manual assembly because ExperimentConfig holds them.
    top = {k: v for k, v in raw.items() if k not in {"data", "depth", "split", "train"}}
    cfg = _build(ExperimentConfig, top)
    if "data" in raw:
        cfg.data = _build(DataConfig, raw["data"])
    if "depth" in raw:
        cfg.depth = _build(DepthConfig, raw["depth"])
    if "split" in raw:
        cfg.split = _build(SplitConfig, raw["split"])
    if "train" in raw:
        cfg.train = _build(TrainConfig, raw["train"])
    return cfg


def config_to_dict(cfg: ExperimentConfig) -> dict:
    return asdict(cfg)
