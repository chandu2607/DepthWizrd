import os
import random
import zipfile
import csv
import sys
import numpy as np
import time
from pathlib import Path
import psutil
try:
    import torch
except ImportError:
    torch = None

ZIP_PATH = r"C:\Users\chand\Downloads\depthwizard_dfc2023_probe\train.zip"
OUT_DIR = Path("data/dfc2023_multicity")
RUN_DIR = Path("runs/dfc2023_multicity_prep")
CACHE_DIR = OUT_DIR / "depth_cache"

RUN_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "rgb").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "dsm").mkdir(parents=True, exist_ok=True)

CITIES_ARM_A = {"Barcelona": 50, "Berlin": 50, "Brasilia": 50, "NewDelhi": 50, "Rio": 50}
CITIES_ARM_B = {
    "Barcelona": 90, "Berlin": 150, "Brasilia": 150, "NewDelhi": 131,
    "Portsmouth": 151, "Rio": 93, "SanDiego": 108, "SaoLuis": 30, "Sydney": 34
}
CITY_VAL = {"Copenhagen": 216}
CITY_TEST = {"NewYork": 108}

def extract_city(filename):
    for city in list(CITIES_ARM_B.keys()) + list(CITY_VAL.keys()) + list(CITY_TEST.keys()):
        if city in filename:
            return city
    return None

def main():
    print("Scanning ZIP...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        namelist = z.namelist()
    
    rgb_files = [f for f in namelist if f.startswith('rgb/') and f.endswith('.tif')]
    
    city_tiles = {}
    for f in rgb_files:
        basename = os.path.basename(f)
        city = extract_city(basename)
        if city:
            city_tiles.setdefault(city, []).append(basename)
    
    random.seed(1337)
    for city in city_tiles:
        city_tiles[city] = sorted(city_tiles[city]) # deterministic sort before sample
        random.shuffle(city_tiles[city])
    
    arm_b_tiles = []
    arm_a_tiles = []
    val_tiles = []
    test_tiles = []
    
    for city, count in CITIES_ARM_B.items():
        if count > len(city_tiles[city]):
            print(f"ERROR: {city} needs {count} but has {len(city_tiles[city])}")
            sys.exit(1)
        selected = city_tiles[city][:count]
        arm_b_tiles.extend([(city, t) for t in selected])
        
        if city in CITIES_ARM_A:
            arm_a_count = CITIES_ARM_A[city]
            arm_a_selected = selected[:arm_a_count]
            arm_a_tiles.extend([(city, t) for t in arm_a_selected])
            
    for city, count in CITY_VAL.items():
        val_selected = city_tiles[city][:count]
        val_tiles.extend([(city, t) for t in val_selected])
        
    for city, count in CITY_TEST.items():
        test_selected = city_tiles[city][:count]
        test_tiles.extend([(city, t) for t in test_selected])

    arm_b_set = set([t for c, t in arm_b_tiles])
    arm_a_set = set([t for c, t in arm_a_tiles])
    val_set = set([t for c, t in val_tiles])
    test_set = set([t for c, t in test_tiles])
    
    print("Verifying...")
    assert len(arm_a_tiles) == 250, f"Arm A has {len(arm_a_tiles)}"
    assert len(arm_b_tiles) == 937, f"Arm B has {len(arm_b_tiles)}"
    assert arm_a_set.issubset(arm_b_set), "Arm A is not subset of Arm B"
    assert len(arm_b_set.intersection(val_set)) == 0
    assert len(arm_b_set.intersection(test_set)) == 0
    assert len(val_set.intersection(test_set)) == 0
    
    all_selected = arm_b_set.union(val_set).union(test_set)
    assert len(all_selected) == 1261, f"Total unique is {len(all_selected)}"
    print("Verification passed.")

    manifest_path = RUN_DIR / "split_manifest.csv"
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['tile_id', 'city', 'split', 'rgb_path', 'dsm_path', 'train_arm_a'])
        
        for city, t in arm_b_tiles:
            writer.writerow([t, city, 'train', f'rgb/{t}', f'dsm/{t}', 'yes' if t in arm_a_set else 'no'])
        for city, t in val_tiles:
            writer.writerow([t, city, 'val', f'rgb/{t}', f'dsm/{t}', 'no'])
        for city, t in test_tiles:
            writer.writerow([t, city, 'test', f'rgb/{t}', f'dsm/{t}', 'no'])
    
    print("Manifest created.")

    print("Extracting files...")
    extracted_rgb = 0
    extracted_dsm = 0
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        for t in all_selected:
            rgb_src = f"rgb/{t}"
            dsm_src = f"dsm/{t}"
            
            z.extract(rgb_src, OUT_DIR)
            extracted_rgb += 1
            z.extract(dsm_src, OUT_DIR)
            extracted_dsm += 1
            
    print(f"Extracted {extracted_rgb} RGB and {extracted_dsm} DSM files.")

    print("Running Smoke Test for Depth Anything...")
    sys.path.append(os.path.abspath("."))
    from depthwizard.depth.depth_anything import DepthAnythingV2
    from depthwizard.config import DepthConfig
    import cv2
    
    cfg = DepthConfig()
    cfg.cache_dir = str(CACHE_DIR)
    
    # Pre-flight resource check
    disk_usage = psutil.disk_usage(str(CACHE_DIR.resolve().parent))
    avail_gb = disk_usage.free / (1024**3)
    est_cache_gb = (1261 * 518 * 518 * 4) / (1024**3) # approx if saved raw uncompressed size is 1MB, compressed is less.
    print(f"Available Disk: {avail_gb:.2f} GB")
    print(f"Estimated Cache: {est_cache_gb:.2f} GB max")
    if avail_gb < est_cache_gb * 2:
        print("ERROR: Insufficient disk space.")
        sys.exit(1)
        
    depth_model = DepthAnythingV2(model_id=cfg.model_id, input_size=cfg.input_size, cache_dir=cfg.cache_dir, use_cache=True)
    
    smoke_tile = list(all_selected)[0]
    rgb_img = cv2.imread(str(OUT_DIR / "rgb" / smoke_tile))
    if rgb_img is None:
        print(f"Failed to read {smoke_tile}")
        sys.exit(1)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    
    print("Smoke inference...")
    t0 = time.time()
    try:
        out = depth_model.infer(rgb_img, smoke_tile, target_hw=rgb_img.shape[:2])
        if torch is not None and torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
            print(f"Peak VRAM: {vram_mb:.2f} MB")
        print(f"Smoke test output shape: {out.shape}, range: {out.min():.2f} - {out.max():.2f}")
        assert not np.isnan(out).any(), "NaN found"
        assert not np.isinf(out).any(), "Inf found"
    except Exception as e:
        print(f"Smoke test failed: {e}")
        sys.exit(1)
    
    print("Smoke test passed. Starting full cache generation...")
    
    missing = 0
    corrupt = 0
    nans = 0
    
    all_selected_list = list(all_selected)
    t_start = time.time()
    for i, t in enumerate(all_selected_list):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(all_selected_list)}")
        
        rgb_path = OUT_DIR / "rgb" / t
        rgb_img = cv2.imread(str(rgb_path))
        if rgb_img is None:
            missing += 1
            continue
            
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        try:
            out = depth_model.infer(rgb_img, t, target_hw=rgb_img.shape[:2])
            if np.isnan(out).any() or np.isinf(out).any():
                nans += 1
        except Exception:
            corrupt += 1
            
    t_end = time.time()
    runtime = t_end - t_start
    
    # Cache validation
    cached_files = list(CACHE_DIR.glob("*.npy"))
    cache_count = len(cached_files)
    total_size_mb = sum(f.stat().st_size for f in cached_files) / (1024**2)
    
    print("\n=== Validation ===")
    print(f"Expected cache files: {len(all_selected_list)}")
    print(f"Actual cache files: {cache_count}")
    print(f"Cache size: {total_size_mb:.2f} MB")
    print(f"Runtime: {runtime:.2f}s ({(runtime/len(all_selected_list)):.2f}s/img)")
    
    with open(RUN_DIR / "preparation_report.md", "w") as f:
        f.write("# Data Preparation Report\n\n")
        f.write(f"## Manifest\n- Arm A count: {len(arm_a_tiles)}\n- Arm B count: {len(arm_b_tiles)}\n")
        f.write(f"- Validation count: {len(val_tiles)}\n- Test count: {len(test_tiles)}\n")
        f.write(f"- Total unique count: {len(all_selected)}\n\n")
        f.write(f"## Extraction\n- RGB extracted: {extracted_rgb}\n- DSM extracted: {extracted_dsm}\n")
        f.write(f"- Missing: {missing}\n\n")
        f.write(f"## Depth Cache\n- Cache count: {cache_count}\n- Size: {total_size_mb:.2f} MB\n")
        f.write(f"- Runtime: {runtime:.2f} s\n")
        if torch is not None and torch.cuda.is_available():
            f.write(f"- Peak VRAM: {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB\n")
        f.write(f"\n## Validation\n- Missing outputs: {missing}\n- Corrupt outputs: {corrupt}\n- NaN/Inf outputs: {nans}\n")

if __name__ == "__main__":
    main()
