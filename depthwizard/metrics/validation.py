"""
Comprehensive Validation & Error Analysis Suite for DepthWizard.
Computes MAE, RMSE, Pearson R, Spearman Rho, Bias, Median AE, P90 AE,
and height-binned accuracy breakdown (<10m, 10-20m, 20-30m, 30-40m, >=40m).
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

class ValidationReport:
    def __init__(
        self,
        summary_metrics: Dict[str, Any],
        binned_table: pd.DataFrame,
        error_map: np.ndarray,
        abs_error_map: np.ndarray
    ):
        self.summary_metrics = summary_metrics
        self.binned_table = binned_table
        self.error_map = error_map
        self.abs_error_map = abs_error_map


def run_validation(
    dsm_pred: np.ndarray,
    dsm_truth: np.ndarray,
    mask_valid: np.ndarray = None
) -> ValidationReport:
    """
    Run full metric validation comparing estimated DSM against ground truth DSM.
    """
    pred = dsm_pred.astype(np.float64)
    truth = dsm_truth.astype(np.float64)
    
    if mask_valid is None:
        mask_valid = (truth > -500.0) & ~np.isnan(truth) & ~np.isnan(pred)
        
    y_pred = pred[mask_valid]
    y_true = truth[mask_valid]
    
    error = pred - truth
    abs_error = np.abs(error)
    
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    bias = float(np.mean(y_pred - y_true))
    med_ae = float(np.median(np.abs(y_true - y_pred)))
    p90_ae = float(np.percentile(np.abs(y_true - y_pred), 90))
    p95_ae = float(np.percentile(np.abs(y_true - y_pred), 95))
    
    r_p, _ = pearsonr(y_true, y_pred) if len(y_true) > 1 else (0.0, 0.0)
    r_s, _ = spearmanr(y_true, y_pred) if len(y_true) > 1 else (0.0, 0.0)
    
    summary = {
        "MAE (m)": round(mae, 2),
        "RMSE (m)": round(rmse, 2),
        "Bias (m)": round(bias, 2),
        "Median AE (m)": round(med_ae, 2),
        "P90 AE (m)": round(p90_ae, 2),
        "P95 AE (m)": round(p95_ae, 2),
        "Pearson R": round(float(r_p), 3),
        "Spearman Rho": round(float(r_s), 3),
        "Valid Pixels": int(mask_valid.sum())
    }
    
    # Binned Height Breakdown
    bins = [
        ("<10 m", 0.0, 10.0),
        ("10–20 m", 10.0, 20.0),
        ("20–30 m", 20.0, 30.0),
        ("30–40 m", 30.0, 40.0),
        ("≥40 m (Tall)", 40.0, 9999.0)
    ]
    
    binned_rows = []
    for label, lo, hi in bins:
        b_mask = (y_true >= lo) & (y_true < hi)
        n_pts = int(b_mask.sum())
        if n_pts > 0:
            b_mae = float(np.mean(np.abs(y_true[b_mask] - y_pred[b_mask])))
            b_bias = float(np.mean(y_pred[b_mask] - y_true[b_mask]))
            b_true_mean = float(np.mean(y_true[b_mask]))
            b_pred_mean = float(np.mean(y_pred[b_mask]))
            binned_rows.append({
                "Height Range": label,
                "Pixel Count": n_pts,
                "True Mean (m)": round(b_true_mean, 1),
                "Pred Mean (m)": round(b_pred_mean, 1),
                "MAE (m)": round(b_mae, 2),
                "Bias (m)": round(b_bias, 2)
            })
            
    df_binned = pd.DataFrame(binned_rows)
    
    return ValidationReport(
        summary_metrics=summary,
        binned_table=df_binned,
        error_map=error.astype(np.float32),
        abs_error_map=abs_error.astype(np.float32)
    )
