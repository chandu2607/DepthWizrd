import tifffile
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/dfc2023_multicity")
manifest_path = Path("runs/dfc2023_multicity_prep/split_manifest.csv")
df = pd.read_csv(manifest_path)

cities = df['city'].unique()
print("Auditing horizontal pixel scales per city:")
for city in sorted(cities):
    city_df = df[df['city'] == city]
    tids = city_df['tile_id'].tolist()
    scales = set()
    projections = set()
    # check first 5 tiles of each city to see if they are consistent
    for tid in tids[:5]:
        rgb_path = DATA_DIR / "rgb" / tid
        with tifffile.TiffFile(rgb_path) as tif:
            page = tif.pages[0]
            scale = page.tags.get(33550) # ModelPixelScaleTag
            scale_val = scale.value if scale else None
            proj = page.tags.get(34737) # GeoAsciiParamsTag
            proj_val = proj.value if proj else None
            
            if scale_val:
                scales.add(tuple(scale_val))
            if proj_val:
                projections.add(proj_val)
                
    print(f"  {city:12s}: pixel scale = {scales}, projections = {projections}")
