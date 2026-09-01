import os
import shutil

brain_dir = r"C:\Users\chand\.gemini\antigravity-ide\brain\2f7ed7b1-86c1-42a7-ae2e-95bfc8a18e0e"
phase47_dir = r"c:\Users\chand\OneDrive\Desktop\DepthWizard\runs\phase47_live_3d_acceptance"

# Copy images to brain dir
images = [
    "01_input_rgb.png",
    "06_final_footprints.png",
    "10_roofs.png",
    "11_walls.png",
    "13_combined_geometry.png",
    "14_final_rgb_city.png",
    "15_target_vs_production.png"
]

for img in images:
    src = os.path.join(phase47_dir, img)
    dst = os.path.join(brain_dir, img)
    if os.path.exists(src):
        shutil.copy(src, dst)

md_content = """# Phase 48 — Visual Evidence Audit

Please review the following generated images to answer the 8 critical human visual test questions.

## 1. Target vs Current Benchmark
![15_target_vs_production.png](file:///C:/Users/chand/.gemini/antigravity-ide/brain/2f7ed7b1-86c1-42a7-ae2e-95bfc8a18e0e/15_target_vs_production.png)

## 2. Final RGB City Render
![14_final_rgb_city.png](file:///C:/Users/chand/.gemini/antigravity-ide/brain/2f7ed7b1-86c1-42a7-ae2e-95bfc8a18e0e/14_final_rgb_city.png)

## 3. Untextured Clay Geometry
![13_combined_geometry.png](file:///C:/Users/chand/.gemini/antigravity-ide/brain/2f7ed7b1-86c1-42a7-ae2e-95bfc8a18e0e/13_combined_geometry.png)

## 4. Footprints
![06_final_footprints.png](file:///C:/Users/chand/.gemini/antigravity-ide/brain/2f7ed7b1-86c1-42a7-ae2e-95bfc8a18e0e/06_final_footprints.png)

## 5. Roofs
![10_roofs.png](file:///C:/Users/chand/.gemini/antigravity-ide/brain/2f7ed7b1-86c1-42a7-ae2e-95bfc8a18e0e/10_roofs.png)

## 6. Walls
![11_walls.png](file:///C:/Users/chand/.gemini/antigravity-ide/brain/2f7ed7b1-86c1-42a7-ae2e-95bfc8a18e0e/11_walls.png)
"""

with open(os.path.join(brain_dir, "phase48_visual_audit.md"), "w") as f:
    f.write(md_content)
    
print("Artifact generated.")
