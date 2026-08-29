import os
import sys
import pandas as pd
import numpy as np
import cv2
from pathlib import Path

# Try to import rasterio or tifffile to inspect metadata
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

DATA_DIR = Path("data/dfc2023_multicity")
manifest_path = Path("runs/dfc2023_multicity_prep/split_manifest.csv")
df = pd.read_csv(manifest_path)

# Let's pick one file per requested city: Berlin, Brasilia, NewDelhi, Copenhagen, NewYork
target_cities = ['Berlin', 'Brasilia', 'NewDelhi', 'Copenhagen', 'NewYork']
selected_tiles = {}

for city in target_cities:
    city_df = df[df['city'] == city]
    if len(city_df) > 0:
        selected_tiles[city] = city_df['tile_id'].iloc[0]

print("Selected tiles for metadata audit:")
for city, tid in selected_tiles.items():
    print(f"  {city}: {tid}")

print("\n--- Metadata Audit ---")
for city, tid in selected_tiles.items():
    rgb_path = DATA_DIR / "rgb" / tid
    dsm_path = DATA_DIR / "dsm" / tid
    
    print(f"\n==================== CITY: {city} (Tile: {tid}) ====================")
    
    # 1. Rasterio Georeferencing Check
    if HAS_RASTERIO:
        print("[rasterio] Reading RGB...")
        try:
            with rasterio.open(rgb_path) as src:
                print("  RGB Profile:")
                print(f"    Driver: {src.driver}")
                print(f"    Width x Height: {src.width} x {src.height}")
                print(f"    CRS: {src.crs}")
                print(f"    Transform (Affine): {src.transform}")
                print(f"    Bounds: {src.bounds}")
                print(f"    Datatype: {src.dtypes}")
                print(f"    Nodata: {src.nodata}")
                print(f"    Tags: {src.tags()}")
                # Check for RPCs
                rpcs = src.rpcs
                print(f"    RPCs present: {rpcs is not None}")
                if rpcs:
                    print("      RPC details:", str(rpcs)[:200])
        except Exception as e:
            print(f"  Error reading RGB: {e}")
            
        print("[rasterio] Reading DSM...")
        try:
            with rasterio.open(dsm_path) as src:
                print("  DSM Profile:")
                print(f"    Driver: {src.driver}")
                print(f"    Width x Height: {src.width} x {src.height}")
                print(f"    CRS: {src.crs}")
                print(f"    Transform (Affine): {src.transform}")
                print(f"    Bounds: {src.bounds}")
                print(f"    Datatype: {src.dtypes}")
                print(f"    Nodata: {src.nodata}")
                print(f"    Tags: {src.tags()}")
        except Exception as e:
            print(f"  Error reading DSM: {e}")
            
    # 2. Tifffile metadata check
    if HAS_TIFFFILE:
        print("[tifffile] Checking TIFF tags...")
        try:
            with tifffile.TiffFile(rgb_path) as tif:
                page = tif.pages[0]
                print(f"  TIFF page shape: {page.shape}, dtype: {page.dtype}")
                # Print geo tags
                geotags = page.geotiff
                print(f"  GeoTIFF tags present: {len(geotags) > 0 if geotags else False}")
                if geotags:
                    print("    KeyGeoTagKeys:", list(geotags.keys()))
                # Check for specific tags like ModelPixelScaleTag, ModelTiepointTag, etc.
                for tag in page.tags.values():
                    if 'ModelPixelScale' in tag.name or 'ModelTiepoint' in tag.name or 'GeoKeyDirectory' in tag.name:
                        print(f"    Tag: {tag.name} = {tag.value}")
        except Exception as e:
            print(f"  Error checking TIFF tags: {e}")
            
    # 3. Simple OpenCV check
    print("[opencv] Checking image dimensions and datatype...")
    img = cv2.imread(str(rgb_path))
    if img is not None:
        print(f"  Shape: {img.shape}, dtype: {img.dtype}")
    else:
        print("  Error loading via OpenCV")
