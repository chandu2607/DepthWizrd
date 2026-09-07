import json
import cv2
import numpy as np
import rasterio
from pathlib import Path
import matplotlib.pyplot as plt

def read_obj(path):
    verts = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(verts)

def run_test(region):
    print(f"\n--- Diagnosing {region} ---")
    
    if region == "uttarakhand":
        scene_dir = Path("c:/Users/chand/OneDrive/Desktop/DepthWizard/runs/phase90_indian_3d_scene") / "UTTARAKHAND_SCENE"
    else:
        scene_dir = Path("c:/Users/chand/OneDrive/Desktop/DepthWizard/runs/phase91_valid_himachal_scene") / "HIMACHAL_SCENE"
        
    with open(scene_dir / "scene_metadata.json", 'r') as f:
        meta = json.load(f)
        
    crop = meta.get("raster_crop_window", {"width": 512, "height": 512, "col_off": 0, "row_off": 0})
    w, h = crop["width"], crop["height"]
    print(f"Crop window: W={w}, H={h}")
    
    src_dir = Path("c:/Users/chand/OneDrive/Desktop/DepthWizard/runs/phase68_india_benchmark_ready/ORIGINAL_DATA") / region
    paths = [src_dir / f"{region}_B04.tif", src_dir / f"{region}_B03.tif", src_dir / f"{region}_B02.tif"]
    from rasterio.windows import Window
    col_off = crop.get("col_off", crop.get("column_offset", 0))
    row_off = crop.get("row_off", crop.get("row_offset", 0))
    win = Window(col_off, row_off, w, h)
    
    bands = [rasterio.open(p).read(1, window=win) for p in paths]
    rgb = np.stack(bands, axis=-1).astype(np.float32)
    rgb = np.clip(rgb / 10000.0 * 255.0, 0, 255).astype(np.uint8)
    
    bldg_verts = read_obj(scene_dir / "building_meshes_finite_height.obj")
    terrain_verts = read_obj(scene_dir / "terrain_mesh.obj")
    x_min, x_max = float(terrain_verts[:, 0].min()), float(terrain_verts[:, 0].max())
    y_min, y_max = float(terrain_verts[:, 1].min()), float(terrain_verts[:, 1].max())
    
    mesh_records = meta.get("mesh_records", [])[:3]
    offset = 0
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, (ax, record) in enumerate(zip(axes, mesh_records)):
        vc = int(record["vertices"])
        verts = bldg_verts[offset:offset+vc]
        offset += vc
        
        n = vc // 2
        top = verts[n:]
        
        poly_x = top[:, 0]
        poly_y = top[:, 1]
        
        u = poly_x / max(x_max, 1.0)
        v = 1.0 - (poly_y / max(y_max, 1.0))
        
        px_x = u * w
        px_y = (1.0 - v) * h  # v = 1 - y/ymax => 1-v = y/ymax
        
        c_x = int(px_x.mean())
        c_y = int(px_y.mean())
        rad = 40
        
        y0 = max(0, c_y - rad)
        y1 = min(h, c_y + rad)
        x0 = max(0, c_x - rad)
        x1 = min(w, c_x + rad)
        
        crop_img = rgb[y0:y1, x0:x1].copy()
        
        crop_px = px_x - x0
        crop_py = px_y - y0
        
        pts = np.stack([crop_px, crop_py], axis=1).astype(np.int32)
        cv2.polylines(crop_img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        
        ax.imshow(crop_img)
        ax.set_title(f"Bldg {record['component_id']}")
        
    plt.tight_layout()
    out = f"c:/Users/chand/OneDrive/Desktop/DepthWizard/runs/phase96_viewer_repair/uv_diag_{region}.png"
    plt.savefig(out)

run_test("uttarakhand")
run_test("himachal")
