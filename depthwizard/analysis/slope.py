"""
Terrain & Structural Slope Analysis Module for DepthWizard.
Computes gradient magnitude (degrees, percentage) and aspect.
Distinguishes natural terrain slope from vertical building facades.
"""

from typing import Dict, Any, Tuple
import numpy as np
import cv2

class SlopeAnalysisResult:
    def __init__(
        self,
        slope_deg: np.ndarray,
        slope_pct: np.ndarray,
        aspect_deg: np.ndarray,
        terrain_slope_deg: np.ndarray,
        facade_mask: np.ndarray,
        stats: Dict[str, Any]
    ):
        self.slope_deg = slope_deg                    # Slope in degrees (0-90)
        self.slope_pct = slope_pct                    # Slope percentage (rise / run * 100)
        self.aspect_deg = aspect_deg                  # Aspect direction (0-360 deg from North)
        self.terrain_slope_deg = terrain_slope_deg    # Pure terrain slope with building walls masked out
        self.facade_mask = facade_mask                # True where slope is steep (>45 deg) along building borders
        self.stats = stats


def compute_slope(
    elevation_raster: np.ndarray,
    gsd_x: float = 1.0,
    gsd_y: float = 1.0,
    mask_bldg: np.ndarray = None
) -> SlopeAnalysisResult:
    """
    Calculate high-precision slope and aspect from an elevation raster.
    Uses Sobel gradient operators scaled by pixel Ground Sampling Distance (GSD).
    """
    Z = elevation_raster.astype(np.float64)
    h, w = Z.shape

    # 3x3 Sobel gradients
    dz_dx = cv2.Sobel(Z, cv2.CV_64F, 1, 0, ksize=3) / (8.0 * max(1e-3, gsd_x))
    dz_dy = cv2.Sobel(Z, cv2.CV_64F, 0, 1, ksize=3) / (8.0 * max(1e-3, gsd_y))

    grad_mag = np.sqrt(dz_dx**2 + dz_dy**2)
    slope_rad = np.arctan(grad_mag)
    slope_deg = np.rad2deg(slope_rad).astype(np.float32)
    slope_pct = (grad_mag * 100.0).astype(np.float32)

    # Aspect in degrees: 0 = North, 90 = East, 180 = South, 270 = West
    aspect_rad = np.arctan2(-dz_dx, dz_dy)
    aspect_deg = np.rad2deg(aspect_rad)
    aspect_deg = np.where(aspect_deg < 0, 360.0 + aspect_deg, aspect_deg).astype(np.float32)

    # Separate natural terrain slope from vertical building facades
    if mask_bldg is not None:
        # Building edges have steep vertical steps (>45 deg)
        dilated = cv2.dilate(mask_bldg.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        eroded = cv2.erode(mask_bldg.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        edge_zone = (dilated > 0) & (eroded == 0)
        facade_mask = edge_zone & (slope_deg > 35.0)
        terrain_mask = ~mask_bldg & ~facade_mask
        terrain_slope = np.where(terrain_mask, slope_deg, np.nan)
        mean_terrain_slope = float(np.nanmean(terrain_slope))
        p95_terrain_slope = float(np.nanpercentile(terrain_slope, 95))
    else:
        facade_mask = slope_deg > 45.0
        terrain_slope = np.where(~facade_mask, slope_deg, np.nan)
        mean_terrain_slope = float(np.nanmean(terrain_slope))
        p95_terrain_slope = float(np.nanpercentile(terrain_slope, 95))

    stats = {
        "mean_overall_slope_deg": round(float(slope_deg.mean()), 2),
        "median_slope_deg": round(float(np.median(slope_deg)), 2),
        "max_slope_deg": round(float(slope_deg.max()), 2),
        "mean_terrain_slope_deg": round(mean_terrain_slope, 2),
        "p95_terrain_slope_deg": round(p95_terrain_slope, 2),
        "steep_slope_pct_area": round(float((slope_deg > 25.0).sum() / (h * w) * 100.0), 2)
    }

    return SlopeAnalysisResult(
        slope_deg=slope_deg,
        slope_pct=slope_pct,
        aspect_deg=aspect_deg,
        terrain_slope_deg=np.nan_to_num(terrain_slope, nan=0.0).astype(np.float32),
        facade_mask=facade_mask,
        stats=stats
    )
