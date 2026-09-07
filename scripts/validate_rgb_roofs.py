import numpy as np
from pathlib import Path
from depthwizard.integration.phase89_scene_adapter import load_phase89_scene

scene = load_phase89_scene("uttarakhand")
roofs = scene["roofs"]
colors = roofs.get("rgb_colors", [])
print("Total rgb colors:", len(colors))
if len(colors) > 0:
    print("Contains NaN:", np.isnan(colors).any())
    print("Contains Inf:", np.isinf(colors).any())
    print("Max/Min:", max(colors), min(colors))
else:
    print("NO COLORS FOUND!")

# Check invariant
bldgs = scene["buildings"]
diffs = []
for b in bldgs:
    if b["height_available"]:
        diffs.append(abs((b["z_roof"] - b["z_ground"]) - b["height_m"]))

if diffs:
    print("Max absolute difference:", max(diffs))
else:
    print("No finite height buildings?")
