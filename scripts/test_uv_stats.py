import json
import numpy as np
from pathlib import Path

def read_obj(path):
    verts = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(verts)

def print_stats(region):
    print(f'\n--- {region} ---')
    if region == 'uttarakhand':
        scene_dir = Path('c:/Users/chand/OneDrive/Desktop/DepthWizard/runs/phase90_indian_3d_scene/UTTARAKHAND_SCENE')
    else:
        scene_dir = Path('c:/Users/chand/OneDrive/Desktop/DepthWizard/runs/phase91_valid_himachal_scene/HIMACHAL_SCENE')
        
    with open(scene_dir / 'scene_metadata.json') as f:
        meta = json.load(f)
    crop = meta.get('raster_crop_window', {'width': 512, 'height': 512})
    w, h = crop['width'], crop['height']
    
    bldg_verts = read_obj(scene_dir / 'building_meshes_finite_height.obj')
    terrain_verts = read_obj(scene_dir / 'terrain_mesh.obj')
    x_min, x_max = float(terrain_verts[:, 0].min()), float(terrain_verts[:, 0].max())
    y_min, y_max = float(terrain_verts[:, 1].min()), float(terrain_verts[:, 1].max())
    
    print(f'Terrain: x_max={x_max}, y_max={y_max}')
    
    mesh_records = meta.get('mesh_records', [])[:2]
    offset = 0
    for record in mesh_records:
        vc = int(record['vertices'])
        verts = bldg_verts[offset:offset+vc]
        offset += vc
        n = vc // 2
        top = verts[n:]
        poly_x, poly_y = top[:, 0], top[:, 1]
        
        u = poly_x / max(x_max, 1.0)
        v = 1.0 - (poly_y / max(y_max, 1.0))
        px_x = u * w
        px_y = (1.0 - v) * h
        
        print(f"Bldg {record['component_id']}:")
        print(f"  X: {poly_x.min():.1f} to {poly_x.max():.1f} | Y: {poly_y.min():.1f} to {poly_y.max():.1f}")
        print(f"  U: {u.min():.4f} to {u.max():.4f} | V: {v.min():.4f} to {v.max():.4f}")
        print(f"  Px: {px_x.min():.1f} to {px_x.max():.1f} | Py: {px_y.min():.1f} to {px_y.max():.1f}")
        print(f"  Is U increasing? {poly_x[1] > poly_x[0] and u[1] > u[0]}")

print_stats('uttarakhand')
print_stats('himachal')
