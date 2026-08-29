import tifffile
from pathlib import Path

DATA_DIR = Path("data/dfc2023_multicity")
rgb_file = DATA_DIR / "rgb" / "SV_NewYork_40.7373_-74.0034.tif"
dsm_file = DATA_DIR / "dsm" / "SV_NewYork_40.7373_-74.0034.tif"

def print_tiff_tags(path):
    print(f"\nTags for: {path.name}")
    with tifffile.TiffFile(path) as tif:
        for i, page in enumerate(tif.pages):
            print(f"Page {i}: shape={page.shape}, dtype={page.dtype}")
            print(f"Number of tags: {len(page.tags)}")
            for tag in page.tags.values():
                val_str = str(tag.value)
                if len(val_str) > 100:
                    val_str = val_str[:100] + "..."
                print(f"  Tag {tag.code:5d} ({tag.name}): {val_str}")

print_tiff_tags(rgb_file)
print_tiff_tags(dsm_file)
