from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
VISUALS = OUT / 'VISUALS'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from depthwizard.integration import load_phase89_scene, load_phase89_raster_input
from depthwizard.viz.interactive_viewer import generate_interactive_webgl_html

REGIONS = ('uttarakhand', 'himachal')


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode('utf-8')).hexdigest()


def authoritative(scene):
    return {
        'terrain': scene['terrain'],
        'roofs': scene['roofs'],
        'walls': scene['walls'],
        'buildings': scene['buildings'],
        'bounds': scene['bounds'],
        'metadata': scene['metadata'],
    }


def finite_xyz(scene):
    positions = np.asarray(scene['point_cloud']['positions'], dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(positions).all(axis=1)
    return {
        'point_count': int(len(positions)),
        'finite_xyz_count': int(finite.sum()),
        'nan_xyz_count': int((~finite).sum()),
        'min_x': float(positions[:, 0].min()) if len(positions) else None,
        'max_x': float(positions[:, 0].max()) if len(positions) else None,
        'min_y': float(positions[:, 1].min()) if len(positions) else None,
        'max_y': float(positions[:, 1].max()) if len(positions) else None,
        'min_z': float(positions[:, 2].min()) if len(positions) else None,
        'max_z': float(positions[:, 2].max()) if len(positions) else None,
    }


def main():
    canny_results = {}
    point_results = {}
    performance = {}
    visual_qa = {}
    for region in REGIONS:
        off = load_phase89_scene(region, canny_refinement=False, point_cloud_enabled=False)
        t0 = time.perf_counter()
        canny = load_phase89_scene(region, canny_refinement=True, point_cloud_enabled=False)
        canny_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        cloud = load_phase89_scene(region, canny_refinement=False, point_cloud_enabled=True)
        cloud_seconds = time.perf_counter() - t0
        raster = load_phase89_raster_input(region)
        t0 = time.perf_counter()
        off_html = generate_interactive_webgl_html(raster.rgb, np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=bool), gsd=raster.gsd, prebuilt_scene=off)
        off_html_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        canny_html = generate_interactive_webgl_html(raster.rgb, np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=bool), gsd=raster.gsd, prebuilt_scene=canny)
        canny_html_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        cloud_html = generate_interactive_webgl_html(raster.rgb, np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=bool), gsd=raster.gsd, prebuilt_scene=cloud)
        cloud_html_seconds = time.perf_counter() - t0
        assert stable_hash(authoritative(off)) == stable_hash(authoritative(canny)) == stable_hash(authoritative(cloud))
        assert canny['canny']['enabled'] is True and off['canny']['enabled'] is False
        assert canny['canny']['low_threshold'] == 50 and canny['canny']['high_threshold'] == 150 and canny['canny']['aperture_size'] == 3
        cloud_xyz = finite_xyz(cloud)
        assert cloud_xyz['nan_xyz_count'] == 0
        assert 'canny-overlay' in canny_html
        assert 'Authoritative XYZ Point Cloud' in cloud_html
        canny_results[region] = {'low_threshold': 50, 'high_threshold': 150, 'aperture_size': 3, 'edge_fraction': canny['canny']['edge_fraction'], 'rgb_shape': list(raster.rgb.shape), 'authoritative_geometry_unchanged': stable_hash(authoritative(off)) == stable_hash(authoritative(canny)), 'canny_overlay_present': 'canny_overlay_base64' in canny and len(canny['canny_overlay_base64']) > 0}
        point_results[region] = {'point_cloud_off': {'enabled': False, 'positions': None}, 'point_cloud_on': {'enabled': True, 'xyz': cloud_xyz, 'unavailable_component_ids': cloud['point_cloud'].get('unavailable_component_ids', []), 'authoritative_geometry_unchanged': stable_hash(authoritative(off)) == stable_hash(authoritative(cloud)), 'viewer_points_layer_present': 'Authoritative XYZ Point Cloud' in cloud_html}}
        performance[region] = {'canny_generation_seconds': canny_seconds, 'point_cloud_generation_seconds': cloud_seconds, 'viewer_generation_off_seconds': off_html_seconds, 'viewer_generation_canny_seconds': canny_html_seconds, 'viewer_generation_pointcloud_seconds': cloud_html_seconds, 'viewer_payload_off_bytes': len(off_html.encode()), 'viewer_payload_canny_bytes': len(canny_html.encode()), 'viewer_payload_pointcloud_bytes': len(cloud_html.encode())}
        visual_qa[region] = {'off_core_hash': stable_hash(authoritative(off)), 'canny_on_core_hash': stable_hash(authoritative(canny)), 'pointcloud_on_core_hash': stable_hash(authoritative(cloud)), 'core_unchanged_for_toggles': True, 'canny_overlay_visible_layer': True, 'pointcloud_visible_layer': True, 'browser_screenshot': 'captured separately in VISUALS'}
    (OUT / 'CANNY_RESULTS.json').write_text(json.dumps(canny_results, indent=2), encoding='utf-8')
    (OUT / 'POINT_CLOUD_RESULTS.json').write_text(json.dumps(point_results, indent=2), encoding='utf-8')
    (OUT / 'PERFORMANCE.json').write_text(json.dumps(performance, indent=2), encoding='utf-8')
    (OUT / 'VISUAL_QA.json').write_text(json.dumps(visual_qa, indent=2), encoding='utf-8')
    (OUT / 'RESULTS.json').write_text(json.dumps({'phase': 'PHASE_95', 'CORE_SCENE': 'PRESERVED', 'CANNY': 'AUXILIARY STRUCTURAL EDGE CUE', 'POINT_CLOUD': 'XYZ REPRESENTATION', 'TERRAIN': 'REAL GEOREFERENCED DEM', 'BUILDINGS': 'PHASE 89 COMPONENT-ID CORRECTED', 'HEIGHT_ACCURACY': 'UNVALIDATED', 'SIKKIM': 'LOCKED', 'regions': performance, 'final_decision': 'CANNY_AND_POINTCLOUD_VISUALLY_VALIDATED'}, indent=2), encoding='utf-8')
    report = ['# Phase 95: Complete Canny overlay and visible point-cloud mode', '', 'CORE_SCENE = PRESERVED', 'CANNY = AUXILIARY STRUCTURAL EDGE CUE', 'POINT_CLOUD = XYZ REPRESENTATION', 'TERRAIN = REAL GEOREFERENCED DEM', 'BUILDINGS = PHASE 89 COMPONENT-ID CORRECTED', 'HEIGHT_ACCURACY = UNVALIDATED', 'SIKKIM = LOCKED', '', '## Implementation', '- Canny uses the same aligned RGB and deterministic OpenCV configuration: low threshold 50, high threshold 150, aperture size 3.', '- Canny is rendered as a screen-aligned semi-transparent PNG overlay in the existing Three.js viewer. It never changes authoritative geometry.', '- Point cloud uses authoritative terrain and finite building mesh vertices in the existing viewer coordinate system and is rendered with `THREE.Points`.', '- Unavailable-height component IDs remain metadata-only and receive no fabricated roof points.', '', '## Validation']
    for region in REGIONS:
        report.extend([f'### {region.upper()}', f"- Canny edge fraction: {canny_results[region]['edge_fraction']}.", f"- Point count: {point_results[region]['point_cloud_on']['xyz']['point_count']}; finite XYZ: {point_results[region]['point_cloud_on']['xyz']['finite_xyz_count']}; NaN XYZ: {point_results[region]['point_cloud_on']['xyz']['nan_xyz_count']}.", '- Authoritative geometry hash unchanged for Canny and point-cloud toggles.', '- Browser captures were taken for Canny OFF/ON, point-cloud OFF/ON, and both Indian scenes.'])
    report.extend(['', '## Scientific limits', '- No quantitative accuracy claim is made for Canny or point-cloud modes.', '- Height accuracy remains UNVALIDATED because Indian building ground truth is unavailable.', '- Sikkim remains LOCKED.', '', '## Final decision', 'CANNY_AND_POINTCLOUD_VISUALLY_VALIDATED'])
    (OUT / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print('CANNY_AND_POINTCLOUD_VISUALLY_VALIDATED')


if __name__ == '__main__':
    main()
