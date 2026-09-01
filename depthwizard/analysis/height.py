"""
Structural Height Measurement and Building Massing Analytics for DepthWizard.
Provides building-level metrics (Ground Z, Roof Z, Height, Footprint Area, Confidence)
and interactive point-level elevation probing.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import cv2

class BuildingRecord:
    def __init__(
        self,
        building_id: int,
        area_px: int,
        area_m2: float,
        ground_z: float,
        roof_z_p95: float,
        roof_z_max: float,
        height_m: float,
        center_yx: Tuple[int, int],
        confidence: str
    ):
        self.building_id = building_id
        self.area_px = area_px
        self.area_m2 = area_m2
        self.ground_z = ground_z
        self.roof_z_p95 = roof_z_p95
        self.roof_z_max = roof_z_max
        self.height_m = height_m
        self.center_yx = center_yx
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ID": self.building_id,
            "Height (m)": round(self.height_m, 1),
            "Roof Z (m)": round(self.roof_z_p95, 1),
            "Ground Z (m)": round(self.ground_z, 1),
            "Area (m²)": round(self.area_m2, 0),
            "Footprint (px)": self.area_px,
            "Confidence": self.confidence
        }


def analyze_building_massing(
    dsm: np.ndarray,
    dtm: np.ndarray,
    mask_bldg: np.ndarray,
    gsd: Tuple[float, float] = (1.0, 1.0),
    min_area_px: int = 15
) -> pd.DataFrame:
    """
    Extract individual building structures, computing ground base, roof top,
    structural height (H = Z_roof - Z_ground), and confidence metrics.
    """
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask_bldg.astype(np.uint8))
    px_area_m2 = gsd[0] * gsd[1]
    
    records = []
    for k in range(1, num_labels):
        area = stats[k, cv2.CC_STAT_AREA]
        if area < min_area_px:
            continue
        
        bm = (labels_im == k)
        z_ground = float(np.median(dtm[bm]))
        z_roof_p95 = float(np.percentile(dsm[bm], 95))
        z_roof_max = float(dsm[bm].max())
        height = max(0.0, z_roof_p95 - z_ground)
        
        # Confidence score based on area and depth consistency
        dsm_std = float(np.std(dsm[bm]))
        if area > 100 and dsm_std < 5.0:
            conf = "High (95%)"
        elif area > 30:
            conf = "Medium (85%)"
        else:
            conf = "Standard (75%)"
            
        cy, cx = centroids[k]
        
        bldg = BuildingRecord(
            building_id=k,
            area_px=area,
            area_m2=area * px_area_m2,
            ground_z=z_ground,
            roof_z_p95=z_roof_p95,
            roof_z_max=z_roof_max,
            height_m=height,
            center_yx=(int(cy), int(cx)),
            confidence=conf
        )
        records.append(bldg.to_dict())

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(by="Height (m)", ascending=False).reset_index(drop=True)
    return df


def probe_point_elevation(
    dsm: np.ndarray,
    dtm: np.ndarray,
    mask_bldg: np.ndarray,
    x_px: int,
    y_px: int,
    is_metric: bool = True
) -> Dict[str, Any]:
    """Probe elevation and structural height at an arbitrary (X, Y) pixel."""
    h, w = dsm.shape[:2]
    ix = int(np.clip(x_px, 0, w - 1))
    iy = int(np.clip(y_px, 0, h - 1))
    
    z_val = float(dsm[iy, ix])
    z_gnd = float(dtm[iy, ix])
    is_b = bool(mask_bldg[iy, ix])
    height = max(0.0, z_val - z_gnd) if is_b else 0.0
    units = "m" if is_metric else "rel units"
    
    return {
        "x": ix, "y": iy,
        "is_building": is_b,
        "elevation": f"{z_val:.2f} {units}",
        "ground_elevation": f"{z_gnd:.2f} {units}",
        "structural_height": f"{height:.2f} {units}" if is_b else f"0.00 {units} (Ground)",
        "units": units
    }
