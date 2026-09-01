import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json

PHASE47_DIR = Path("runs/phase47_live_3d_acceptance")
OUT_DIR = Path("runs/phase48_visual_audit")
TRACES_DIR = OUT_DIR / "building_traces"

def main():
    print("PHASE 48: VISUAL EVIDENCE AUDIT", flush=True)
    
    # Load images
    img_rgb = cv2.imread(str(PHASE47_DIR / "01_input_rgb.png"))
    img_footprint = cv2.imread(str(PHASE47_DIR / "04_building_mask.png"), cv2.IMREAD_GRAYSCALE)
    img_roof = cv2.imread(str(PHASE47_DIR / "10_roofs.png"), cv2.IMREAD_GRAYSCALE)
    img_wall = cv2.imread(str(PHASE47_DIR / "11_walls.png"), cv2.IMREAD_GRAYSCALE)
    
    if img_rgb is None or img_footprint is None:
        print("Error loading images!")
        return

    _, img_footprint = cv2.threshold(img_footprint, 127, 255, cv2.THRESH_BINARY)
    _, img_roof = cv2.threshold(img_roof, 127, 255, cv2.THRESH_BINARY)
    _, img_wall = cv2.threshold(img_wall, 127, 255, cv2.THRESH_BINARY)

    # Extract components
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_footprint, connectivity=8)
    
    valid_buildings = []
    for b_id in range(1, n_labels):
        area = stats[b_id, cv2.CC_STAT_AREA]
        if area > 100:
            valid_buildings.append(b_id)
            
    # Select 10 buildings (sorted by area descending)
    valid_buildings = sorted(valid_buildings, key=lambda x: stats[x, cv2.CC_STAT_AREA], reverse=True)[:10]
    
    audit_data = []
    
    for idx, b_id in enumerate(valid_buildings):
        x, y, w, h, fp_area = stats[b_id]
        
        # Crop region with padding
        pad = 20
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(img_rgb.shape[1], x + w + pad), min(img_rgb.shape[0], y + h + pad)
        
        b_mask = (labels == b_id).astype(np.uint8) * 255
        
        # Calculate roof inside this footprint
        roof_in_fp = cv2.bitwise_and(img_roof, b_mask)
        roof_area = cv2.countNonZero(roof_in_fp)
        
        # Find reasons for area loss
        diff_mask = cv2.bitwise_xor(b_mask, roof_in_fp)
        erosion_loss = 0
        
        # simplified check: is loss at boundaries?
        eroded_fp = cv2.erode(b_mask, np.ones((3,3), np.uint8), iterations=1)
        boundary_pixels = cv2.bitwise_xor(b_mask, eroded_fp)
        loss_at_boundary = cv2.bitwise_and(diff_mask, boundary_pixels)
        erosion_loss = cv2.countNonZero(loss_at_boundary)
        
        # Create traces
        crop_rgb = img_rgb[y1:y2, x1:x2]
        crop_fp = b_mask[y1:y2, x1:x2]
        crop_roof = img_roof[y1:y2, x1:x2]
        crop_wall = img_wall[y1:y2, x1:x2]
        
        cv2.imwrite(str(TRACES_DIR / f"b{idx:02d}_rgb.png"), crop_rgb)
        cv2.imwrite(str(TRACES_DIR / f"b{idx:02d}_footprint.png"), crop_fp)
        cv2.imwrite(str(TRACES_DIR / f"b{idx:02d}_roof.png"), crop_roof)
        cv2.imwrite(str(TRACES_DIR / f"b{idx:02d}_wall.png"), crop_wall)
        
        audit_data.append({
            "id": idx,
            "fp_area": int(fp_area),
            "roof_area": int(roof_area),
            "ratio": round(roof_area / max(1, fp_area), 4),
            "loss_pixels": int(fp_area - roof_area),
            "boundary_loss_pixels": int(erosion_loss)
        })

    df = pd.DataFrame(audit_data)
    print("\n--- CRITICAL ROOF TEST ---")
    print(df.to_string(index=False))
    
    total_fp = df["fp_area"].sum()
    total_roof = df["roof_area"].sum()
    total_boundary_loss = df["boundary_loss_pixels"].sum()
    total_loss = total_fp - total_roof
    
    print("\nMissing 9.7% Analysis:")
    print(f"Total Loss: {total_loss} pixels")
    print(f"Loss at Boundary (Erosion/Simplification): {total_boundary_loss} pixels ({total_boundary_loss/max(1, total_loss)*100:.1f}%)")
    
    with open(OUT_DIR / "audit_stats.json", "w") as f:
        json.dump(audit_data, f, indent=4)
        
if __name__ == "__main__":
    main()
