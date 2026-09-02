# PHASE 14B — FULL DECODER-ADAPTATION EXPERIMENT

This report details the results of fine-tuning the DPT decoder (neck and head, ~2.7M parameters) of Depth Anything V2 while keeping the ViT backbone completely frozen. The goal was to test if task-specific adaptation of the decoder could break the ~14m prediction ceiling observed in previous experiments.

## 1. Experiment Setup
- **Model:** `Depth Anything V2 Small` (HuggingFace `AutoModelForDepthEstimation`)
- **Trainable Parameters:** 2,728,513 (Decoder Neck + Head)
- **Frozen Parameters:** 22,056,576 (ViT Backbone)
- **Target:** `log1p(height)`
- **Loss:** Standard masked-L1
- **Seeds:** 2 (Seed 0, Seed 1)
- **Data:** Trained on Arm-B (9 cities), validated on Copenhagen, tested on New York.

## 2. Resource Utilization
- **Peak VRAM:** 793.4 MB (Batch size 4)
- **Runtime:** ~15 minutes per seed for 15 epochs.

## 3. Results (Seed 0)

### Overall Metrics
- **MAE:** 8.89m
- **RMSE:** 16.04m
- **Pearson r:** 0.624

### Binned Height Metrics (The Ceiling Check)
| Height Bin | MAE (m) | Bias (m) |
|------------|---------|----------|
| 0–2 m      | 1.91    | +1.89    |
| 2–5 m      | 5.80    | +2.39    |
| 5–10 m     | 6.18    | -0.60    |
| 10–15 m    | 6.48    | -5.12    |
| 15–20 m    | 8.67    | -8.17    |
| 20–30 m    | 12.55   | -12.41   |
| 30–40 m    | 21.40   | -21.37   |
| >40 m      | 39.16   | -39.16   |

*Note: Seed 1 produced nearly identical results (Overall MAE 8.84m, >40m Bias -39.70m).*

## 4. Ceiling Analysis
The tall-height prediction ceiling is completely intact. 
- For buildings >40m tall, the model under-predicts by an average of -39.16m (meaning it predicts the ground). 
- For buildings 30-40m tall, it under-predicts by -21.37m, yielding an average prediction of ~13.5m.
- This is the exact same ~14m ceiling exhibited by the original frozen `C_log1p` baseline.

## 5. Scientific Interpretation
**CASE C — NO SUPPORT.**

The tall collapse remains essentially unchanged. 
Decoder adaptation is not enough. The 2.7M trainable parameters in the DPT neck and head are unable to synthesize missing metric-scale features. The limitation lies deeper in the frozen ViT representation itself (which is oblivious to absolute camera/metric scale). Without unfreezing the backbone to allow gradients to re-calibrate the core attention maps, the decoder can only act on the scale-ambiguous relative depth features it is fed, defaulting safely back to the dominant low-rise distribution.
