"""
Unified Raster Ingestion & Metadata Parser for DepthWizard.
Supports: PNG, JPG, JPEG, TIFF, GeoTIFF.
Automatically extracts CRS, Affine Transform, GSD, and Bounds when georeferencing is present.
"""

import io
import os
import tempfile
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2
import rasterio
import rasterio.transform

class RasterInput:
    """Represents an ingested raster image with verified metadata."""
    def __init__(
        self,
        rgb: np.ndarray,
        filename: str,
        is_georeferenced: bool,
        crs: Optional[str] = None,
        transform: Optional[rasterio.transform.Affine] = None,
        bounds: Optional[Any] = None,
        gsd: Optional[Tuple[float, float]] = None,
        raw_bytes: Optional[bytes] = None
    ):
        self.rgb = rgb  # uint8 (H, W, 3) in RGB
        self.filename = filename
        self.is_georeferenced = is_georeferenced
        self.crs = crs
        self.transform = transform if transform is not None else rasterio.transform.from_origin(0, rgb.shape[0], 1.0, 1.0)
        self.bounds = bounds
        self.gsd = gsd if gsd is not None else (1.0, 1.0)
        self.raw_bytes = raw_bytes

    @property
    def shape(self) -> Tuple[int, int]:
        return self.rgb.shape[:2]

    @property
    def height(self) -> int:
        return self.rgb.shape[0]

    @property
    def width(self) -> int:
        return self.rgb.shape[1]

    def get_summary(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "dimensions": f"{self.width} x {self.height} px",
            "is_georeferenced": self.is_georeferenced,
            "mode": "ABSOLUTE METRIC DSM" if self.is_georeferenced else "RELATIVE rDSM",
            "crs": self.crs if self.crs else "Non-georeferenced (Relative)",
            "gsd_m": f"{self.gsd[0]:.3f} m x {self.gsd[1]:.3f} m" if self.is_georeferenced else "N/A (Pixel Units)",
            "bounds": [round(b, 2) for b in (self.bounds.left, self.bounds.bottom, self.bounds.right, self.bounds.top)] if self.bounds else None
        }


def load_raster_input(path_or_bytes: Any, filename: str = "input_raster.tif") -> RasterInput:
    """
    Load any supported raster (PNG, JPG, TIFF, GeoTIFF) from path or raw bytes.
    Returns a unified RasterInput instance.
    """
    is_bytes = isinstance(path_or_bytes, (bytes, io.BytesIO))
    raw_data = path_or_bytes if isinstance(path_or_bytes, bytes) else (path_or_bytes.getvalue() if isinstance(path_or_bytes, io.BytesIO) else None)

    # First attempt: rasterio reader for geospatial metadata
    try:
        if is_bytes:
            ctx = rasterio.open(io.BytesIO(raw_data))
        else:
            ctx = rasterio.open(path_or_bytes)

        with ctx as src:
            count = src.count
            if count >= 3:
                bands = src.read([1, 2, 3])
            elif count == 1:
                b = src.read(1)
                bands = np.stack([b, b, b])
            else:
                b1 = src.read(1)
                bands = np.stack([b1, b1, b1])

            def _to_u8(arr):
                mn, mx = float(arr.min()), float(arr.max())
                if mx > mn:
                    return ((arr - mn) / (mx - mn + 1e-6) * 255.0).astype(np.uint8)
                return np.zeros_like(arr, dtype=np.uint8)

            rgb_img = np.transpose(np.stack([_to_u8(bands[i]) for i in range(3)]), (1, 2, 0))
            is_geo = (src.crs is not None) and (src.transform is not None)
            crs_str = str(src.crs) if is_geo else None
            trans = src.transform if is_geo else None
            bounds = src.bounds if is_geo else None
            gsd = (abs(src.transform.a), abs(src.transform.e)) if is_geo else None

            return RasterInput(
                rgb=rgb_img,
                filename=filename,
                is_georeferenced=is_geo,
                crs=crs_str,
                transform=trans,
                bounds=bounds,
                gsd=gsd,
                raw_bytes=raw_data
            )
    except Exception:
        pass

    # Second attempt: standard image decoder (PNG, JPG, BMP)
    if is_bytes:
        arr = np.frombuffer(raw_data, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        decoded = cv2.imread(str(path_or_bytes), cv2.IMREAD_COLOR)

    if decoded is None:
        raise ValueError(f"Could not decode image from {filename}. Supported formats: PNG, JPG, JPEG, TIFF, GeoTIFF.")

    rgb_img = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return RasterInput(
        rgb=rgb_img,
        filename=filename,
        is_georeferenced=False,
        crs=None,
        transform=rasterio.transform.from_origin(0, rgb_img.shape[0], 1.0, 1.0),
        bounds=None,
        gsd=(1.0, 1.0),
        raw_bytes=raw_data
    )
