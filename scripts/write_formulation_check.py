import os
import json
from pathlib import Path

OUT_DIR = Path("runs/phase19b_oots_formulation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

formulation_md = """# PHASE 19B — OOTS-NET FORMULATION CHECK REPORT

## 1. Normalization Analysis

If we apply **local min-max normalization** to the local relative-depth map $N_{rel, i}$ inside each building component $i$:
$$N_{norm, i} = \\frac{N_{rel, i} - \\min(N_{rel, i})}{\\max(N_{rel, i}) - \\min(N_{rel, i}) + \\epsilon}$$

### What is lost?
This transformation maps all building components to the range $[0, 1]$, which **destroys the absolute relative depth range** (the difference between the maximum and minimum relative depth). For example:
- A flat-roof building (depth range of $0.05$ relative depth units)
- A tall, complex building (depth range of $0.8$ relative depth units)
will **both** normalize to $[0, 1]$ with maximums at $1.0$ and minimums at $0.0$.
This loss is highly detrimental because the raw relative-depth range is a strong indicator of building height. Discarding it removes the vertical structural signature of the building.

### Does local normalization help structure?
**Yes.** It standardizes the structural shape (flat vs. gabled vs. domed) which makes the geometry learnable, but we must **not** throw away the raw range. The optimal solution is to preserve both: the normalized topology for roof shape and the unnormalized relative depth range as an input feature for scale prediction.

---

## 2. Scale Target Analysis

In Phase 18, we compared Max height and P95 height targets:
- **Max Height Target:** Sensitive to single-pixel outliers (sensor glint, orthorectification artifacts at roof edges).
- **P95 Height Target:** Highly stable robust upper quantile that filters out noise while still capturing the structural height of the building.

Therefore, the scale target $S_i$ should represent **P95 building height**, providing a stable anchor for the reconstructed building.

---

## 3. Two Buildings in One Tile: Resolving Scale Collapse

Under the failed tile-level P99 approach, a single scale factor is predicted for the entire tile. If a tile contains:
- Building A (8m)
- Building B (25m)
- Building C (70m)
A single tile-wide scale prediction ($P_{99} \\approx 70$m) scales up Buildings A and B incorrectly.
By predicting **independent building-level scales** $S_i$ for each segmented component:
- Building A is scaled by $S_A \\approx 8$m
- Building B by $S_B \\approx 25$m
- Building C by $S_C \\approx 70$m
This completely avoids global scale collapse and respects local building variation.

---

## 4. Dense nDSM Reconstruction

The dense 512x512 nDSM is reconstructed by rasterizing independent building components:
- **Touching/Merged components:** Touching buildings in the predicted mask will merge into single components. The model will scale the merged component using its combined area and depth. This represents a minor resolution degradation rather than scale collapse.
- **Overlapping Masks:** Resolved because components are extracted from a single predicted binary footprint raster (each pixel is assigned to exactly one component label).
- **Holes/Boundaries:** Interior pixels are scaled via $(S_i - 2.0) \\times N_{norm} + 2.0$, and non-building pixels are set to $0.0$, maintaining sharp structural boundaries.

---

## 5. Tall-Tail Test: Distinguishing 20m vs. 100m Buildings

If a 20m and 100m building both have flat roofs, their normalized maps $N_{norm}$ will be identical flat sheets.
To distinguish them, the scale predictor $S_i$ must use:
1.  **GSD-anchored footprint geometry:** Footprint area ($m^2$) and perimeter.
2.  **Raw local relative-depth range:** The unnormalized depth range ($P_{99} - P_{10}$) within the building mask, which remains much larger for a 100m building than a 20m building due to shadows, parallax, and contrast.

---

## 6. Literature Overlap

OOTS-Net is distinguished from existing works (HTC-DC, HGDNet) by **calibrating relative foundation-model depth predictions at the object level** using GSD-anchored footprint priors, enabling zero-shot cross-city height transfer.

---

## 7. Recommended Formulation

We recommend **Option C: Use both normalized topology + depth-range/statistical features.**

### Reconstruction Formula:
$$h_{pixel} = (S_i - 2.0) \\times N_{norm, i} + 2.0$$
where $S_i$ is the predicted P95 building height, and $N_{norm, i}$ is the locally normalized relative-depth topology.

### Scale Input Features for $S_i$:
- **Geometry (7-D):** Area ($m^2$), bounding box dimensions, aspect ratio, contour perimeter, and isoperimetric compactness.
- **Relative Depth Stats (9-D):** Local relative depth mean, median, P95, standard deviation, and **unnormalized depth range** ($P_{99} - P_{10}$).

---

## 8. Minimal Falsifiable Experiment

**Setup:** Train the Multi-Task U-Net on 128 training tiles. Evaluate the OOTS-Net scale predictor on New York buildings.
**Falsification Threshold:** If the zero-shot P95 height prediction MAE on New York is **$> 32\text{m}$** or the Pearson correlation is **$< 0.20$**, the object-level scale formulation is falsified and must be abandoned.

---

## 9. Final Decision

```text
MODIFY OOTS FORMULATION
```

- **Exact formula:** $h_{pixel} = (S_i - 2.0) \\times \\text{Normalize}(N_{rel, i}) + 2.0$.
- **What is preserved:** Fine relative roof shapes (via min-max normalized topology) and absolute vertical height indicators (via raw relative-depth range).
- **What is deliberately discarded:** Global tile-level average depth statistics, which are highly collinear and domain-sensitive.
- **Technical Risk:** Component merging in dense urban blocks.
"""

results_json = {
  "normalization_analysis": "Local min-max normalization standardizes shape but discards depth range. We must preserve both normalized topology and unnormalized depth range.",
  "recommended_formulation": "Option C: Normalized topology + depth-range/statistical features.",
  "final_decision": "MODIFY OOTS FORMULATION"
}

with open(OUT_DIR / "FORMULATION_REVIEW.md", "w") as f:
    f.write(formulation_md)

with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results_json, f, indent=2)

print("Saved FORMULATION_REVIEW.md and results.json to runs/phase19b_oots_formulation/")
