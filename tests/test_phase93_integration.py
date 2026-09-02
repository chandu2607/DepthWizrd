from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from depthwizard.integration import load_phase89_scene
from depthwizard.viz.interactive_viewer import generate_interactive_webgl_html


def test_phase89_scenes_preserve_component_ids_and_missing_heights():
    for region in ("uttarakhand", "himachal"):
        scene = load_phase89_scene(region)
        buildings = scene["buildings"]
        ids = [building["component_id"] for building in buildings]
        assert len(ids) == len(set(ids))
        assert all(building["height_available"] == (building["height_m"] is not None) for building in buildings)
        assert all(building["height_m"] is not None for building in buildings if building["height_available"])
        assert all(building["height_m"] is None for building in buildings if not building["height_available"])
        assert scene["metadata"]["terrain_source"] == "REAL_GEOREFERENCED_DEM"


def test_phase89_scene_serializes_without_nonfinite_values():
    scene = load_phase89_scene("himachal")
    json.dumps(scene, allow_nan=False)


def test_phase89_scene_canny_and_point_cloud_are_opt_in():
    scene = load_phase89_scene("uttarakhand")
    assert scene["canny"]["enabled"] is False
    assert scene["point_cloud"]["enabled"] is False
    canny_scene = load_phase89_scene("uttarakhand", canny_refinement=True)
    cloud_scene = load_phase89_scene("uttarakhand", point_cloud_enabled=True)
    assert canny_scene["canny"]["enabled"] is True
    assert canny_scene["canny"]["low_threshold"] == 50
    assert canny_scene["canny"]["high_threshold"] == 150
    assert canny_scene["canny"]["aperture_size"] == 3
    assert cloud_scene["point_cloud"]["enabled"] is True
    assert len(cloud_scene["point_cloud"]["positions"]) > 0
    assert cloud_scene["point_cloud"]["unavailable_component_ids"] == [8, 11]


def test_default_viewer_fallback_still_builds_geometry():
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    dsm = np.zeros((32, 32), dtype=np.float32)
    dtm = np.zeros((32, 32), dtype=np.float32)
    mask = np.zeros((32, 32), dtype=bool)
    html = generate_interactive_webgl_html(rgb, dsm, dtm, mask, gsd=1.0, stride=4)
    assert "DepthWizard 3D Interactive City Flythrough" in html
    assert "OrbitControls" in html


def test_prebuilt_viewer_override_uses_authoritative_scene():
    scene = load_phase89_scene("himachal")
    html = generate_interactive_webgl_html(
        np.zeros((32, 32, 3), dtype=np.uint8),
        np.zeros((32, 32), dtype=np.float32),
        np.zeros((32, 32), dtype=np.float32),
        np.zeros((32, 32), dtype=bool),
        prebuilt_scene=scene,
    )
    assert "PHASE 89 COMPONENT-ID CORRECTED" in html
    assert "HEIGHT_UNAVAILABLE" in html
    assert scene["point_cloud"]["enabled"] is False
    cloud_html = generate_interactive_webgl_html(
        np.zeros((32, 32, 3), dtype=np.uint8),
        np.zeros((32, 32), dtype=np.float32),
        np.zeros((32, 32), dtype=np.float32),
        np.zeros((32, 32), dtype=bool),
        prebuilt_scene=load_phase89_scene("uttarakhand", canny_refinement=True, point_cloud_enabled=True),
    )
    assert "Authoritative XYZ Point Cloud" in cloud_html
    assert "canny-overlay" in cloud_html
