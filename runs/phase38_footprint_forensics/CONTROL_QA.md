# DepthWizard — Phase 38 WebGL Controls & Interaction QA Report

| Control Category | Action Tested | Measured Before State | Action Taken | Measured After State | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Camera Presets** | `City Overview` | Camera at `(0, 500, 0)` | Click Preset | Camera moves to `(-1.02*maxDim, 1.35*maxDim, 1.02*maxDim)` framing full block with 20% margin | ✅ PASS |
| | `Urban Oblique` | Camera Overview | Click Preset | Camera moves to `(-0.69*maxDim, 0.495*maxDim, 0.69*maxDim)` 35° facade perspective | ✅ PASS |
| | `Inspection` | Camera Oblique | Click Preset | Camera zooms dynamically to closest distance on tallest building structure | ✅ PASS |
| | `Top-Down` | Camera Inspection | Click Preset | Camera targets nadir 90° overhead perspective `(0, 1.73*maxDim, 0.5)` | ✅ PASS |
| | `Pedestrian` | Camera Top-Down | Click Preset | Camera drops to street-level ground coordinate looking up at city skyline | ✅ PASS |
| **Render Modes** | `RGB City` | Default WebGL shaders | Click Render Mode | Applies satellite orthophoto texture map to terrain & roofs; slate walls | ✅ PASS |
| | `Elevation` | RGB texture mode | Click Render Mode | Applies Turbo elevation colormap to terrain, roofs, walls & shows HUD legend | ✅ PASS |
| | `Building Height` | Elevation mode | Click Render Mode | Subdues terrain and applies Turbo height colormap (0-60m+) to structures | ✅ PASS |
| | `Terrain Slope` | Height mode | Click Render Mode | Applies slope colormap (0° Green to 45°+ Red) to ground DTM grid | ✅ PASS |
| **Navigation** | Orbit / Pan / Zoom | Static viewport | Mouse Drag / Scroll | Camera matrix updates via OrbitControls with smooth damping | ✅ PASS |
| | First-Person Fly | Default view | Press WASD / Arrows | Camera position vectors update smoothly in ground plane & vertical axes | ✅ PASS |
| | `Cinematic Flythrough` | Static view | Click Flythrough Toggle | Continuous 360° orbiting animation loop triggers at 60 FPS | ✅ PASS |
| **Inspector HUD** | Raycast Building Pick | Inspector panel hidden | Click building roof/wall | Inspector HUD opens displaying ID, Roof Z, Ground Z, Height m, Area m² | ✅ PASS |
| **Exaggeration** | Z-scale slider | Exaggeration 1.0x | Slide to 2.0x | WebGL geometry Z positions double without modifying underlying scientific heights | ✅ PASS |
