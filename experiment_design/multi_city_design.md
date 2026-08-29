# DepthWizard Multi-City Experiment Design

## Objective
To determine if training on multiple geographically and scenically different cities improves the generalization of metric building-height prediction to completely unseen cities, particularly for tall structures.

## City Split Strategy

Based on the verified DFC2023 dataset statistics, we split the 11 available cities as follows:

| Role | City | Tiles Used (Arm B) | Tall-height coverage (>15m) | Reason |
|------|------|--------------------|-----------------------------|--------|
| Validation | Copenhagen | 216 | 26.31% | Good moderate-to-tall representation, European architecture. Strictly unseen during training. |
| Final Test | NewYork | 108 | 34.55% | Ultimate test for tall, dense structures (15.8% >30m). Completely unseen during training/validation. |
| Train | Barcelona | 90 | 47.72% | Extremely high >15m exposure, provides strong tall-building signal. |
| Train | Berlin | 150 (cap) | 14.55% | Balanced moderate-rise. Capped from 566 to prevent data dominance. |
| Train | Brasilia | 150 (cap) | 0.60% | South American, very low-rise, sparser. Capped from 246. |
| Train | NewDelhi | 131 | 11.43% | Asian, very tall max height (183m). |
| Train | Portsmouth | 151 | 3.01% | European, low/moderate-rise. |
| Train | Rio | 93 | 14.15% | South American, tall (148m max), dense. |
| Train | SanDiego | 108 | 0.00% | North American, strictly low-rise. |
| Train | SaoLuis | 30 | 2.28% | South American, extremely dense building count (321/img). |
| Train | Sydney | 34 | 18.21% | Australian, medium/tall rise. |

### Excluded Cities
None of the 11 cities are discarded, but Berlin (566) and Brasilia (246) are heavily sub-sampled to prevent distribution collapse into a single city's statistics.

## Training Arms

### Arm A — Size-Matched Diversity
**Purpose:** Test if diversity alone matters when keeping the data size matched to the original baseline (~250 tiles).
**Composition:** 250 tiles total.
- 50 tiles from Barcelona
- 50 tiles from Berlin
- 50 tiles from Brasilia
- 50 tiles from NewDelhi
- 50 tiles from Rio
This provides a highly diverse but small dataset representing low, medium, and extremely tall environments across Europe, Asia, and South America.

### Arm B — Larger Multi-City
**Purpose:** Test if additional data *plus* diversity provides further improvement.
**Composition:** 937 tiles total.
- Barcelona (90), NewDelhi (131), Portsmouth (151), Rio (93), SanDiego (108), SaoLuis (30), Sydney (34)
- Berlin (150 random subset)
- Brasilia (150 random subset)

## Leakage Checks
- **Test Set Isolation:** NewYork is exclusively used for the final test set. No NewYork tiles are in Train/Val.
- **Val Set Isolation:** Copenhagen is exclusively used for Val. No Copenhagen tiles are in Train/Test.
- **Geographic Overlap:** All cities are globally distinct; no spatial overlap is possible.
- **Data Deduplication:** Train, Val, and Test are sourced from strictly different city sub-directories in the archive.

## Fixed Variables
To ensure a controlled experiment, the following will remain unchanged from the `C_log1p` baseline:
- **Base Model:** Depth Anything V2 (frozen weights).
- **Architecture:** C_log1p fusion U-Net.
- **Target Transform:** log1p(height).
- **Hyperparameters:** Loss, optimizer, learning rate, batch size, train resolution, width.
- **Data Pipeline:** Augmentation, preprocessing, evaluation metrics, random seeds.

## Evaluation Criteria

### Success Criteria
- Improved unseen-city building MAE and RMSE (on NewYork).
- Improved >15m, >20m, >30m, and >40m MAE on NewYork.
- Improved tall-height signed bias.
- No severe regression in low-height performance.
- No artificial upward shift (predicting everything as taller).
- Consistent results across random seeds.

### Failure Criteria
- Unseen-city (NewYork) tall heights (>30m, >40m) remain highly inaccurate or underestimated.
- Overall MAE improves only by overfitting to low-rise buildings, while the primary bottleneck (tall structures) persists.
- Edge artifacts appear, or the model predicts high structures everywhere.

## Expected Compute & Storage (Cache Generation)
Before training, Depth Anything features must be cached.
- **Total Tiles (Arm B + Val + Test):** 937 + 216 + 108 = 1261 tiles.
- **Compute:** Assuming ~0.5s per tile for DA-v2 inference, caching will take ~10-15 minutes on a modern GPU.
- **Storage:** If caching 14x14x1024 or similar features, expect ~10-20 GB depending on float16/float32 precision and resolution.

## Exact Next Action
1. User approves the experiment design.
2. Filter the `train.zip` payload to extract only the 1261 required tiles (or extract the necessary ones dynamically).
3. Generate the frozen Depth Anything cache for these specific subsets.
4. Run training for Arm A and Arm B.
