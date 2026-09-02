# PHASE 13B - ORDINAL HEIGHT CLASSIFICATION (FULL EXPERIMENT)

## EXPERIMENT SUMMARY
- **Goal:** Test if removing continuous L1 regression pressure by switching to cross-entropy height-bin classification allows the network to recover the tall-height tail.
- **Model:** Depth-only, Frozen Depth Anything V2, SmallFusionUNet (out=8 classes).
- **Target:** 8 ordinal height bins (0-2, 2-5, 5-10, 10-15, 15-20, 20-30, 30-40, >40m).
- **Loss:** Standard Multiclass Cross-Entropy (unweighted).
- **Evaluation:** New York test set. Seed 0 and 1.

## RESULTS: CLASS DISTRIBUTION & COLLAPSE
The experiment yielded an extreme class collapse. Despite the training set containing over 1.9 million pixels in the >40m class, the classifier completely refused to predict the top classes.

**Confusion Matrix (Seed 0)**
Rows = True Class, Columns = Predicted Class
```text
            [P0] [P1] [P2]   [P3] [P4] [P5]   [P6] [P7]
True 0-2m   15.7M  0  67K      2    0   195K    0    0
True 2-5m   319K   0  4K       1    0   7K      0    0
True 5-10m  633K   0  13K      3    2   21K     0    0
True 10-15m 1.45M  0  39K     18    2   31K     0    0
True 15-20m 2.07M  0  71K     79    1   94K     0    0
True 20-30m 2.77M  0  133K   221    9   149K    0    0
True 30-40m 1.35M  0  76K   4805   40   153K    0    0
True >40m   2.41M  0  172K   983    6   301K    0    0
```
- The network almost entirely predicts Class 0 (0-2m), Class 5 (20-30m), and Class 2 (5-10m).
- **Zero pixels** were predicted as Class 6 (30-40m) or Class 7 (>40m) across the entire New York test set.

## QUANTITATIVE METRICS (Mean ± Std over 2 seeds)
- **Overall Accuracy:** 0.558 ± 0.003
- **Macro F1:** 0.102 ± 0.002
- **>15m Recall:** 0.046 ± 0.026
- **>20m Recall:** 0.051 ± 0.029
- **>30m Recall:** 0.000 ± 0.000
- **>40m Recall:** 0.000 ± 0.000
- **>40m Precision:** 0.000 ± 0.000

## COMPARISON AGAINST REGRESSION
- **Old Regression (C_log1p):** Predicted extreme tall buildings as ~30-40m (flattened ceiling).
- **Classification:** Predicted extreme tall buildings predominantly as 0-2m (ground) or 20-30m. It completely failed to allocate *any* spatial footprint to the >40m class.
- **Verdict:** Classification performed categorically *worse* at identifying the extreme tail than regression did.

## SCIENTIFIC VERDICT: NOT SUPPORTED
**The classification hypothesis is NOT SUPPORTED.**
The hypothesis assumed that the unbounded geometric penalty of L1 regression was suppressing the tail, and that a class-based cross-entropy formulation would free the network to predict extreme heights. The data directly falsifies this as the primary cause. Even when the mathematical L1 penalty was completely removed, the network still suffered from a catastrophic height ceiling and failed to recall a single >30m or >40m pixel. 

**Interpretation (Case C):**
The problem is NOT primarily the regression loss formulation. Because the network collapses the tall classes even under cross-entropy, the evidence strongly points back to the **frozen Depth Anything V2 representation itself**. The relative-depth feature map generated from the overhead RGB image simply lacks the unambiguous metric scaling features required to distinguish a 25m building from a 60m building. Without a reliable input signal to separate the classes, the classifier safely defaults to the dominant distribution modes (ground and mid-rise) to minimize expected cross-entropy.
