import json
import numpy as np

with open('runs/phase14e_vit_unfreeze/results.json') as f:
    res = json.load(f)

for r in res:
    print(f"=== SEED {r['seed']} ===")
    am = r['agg_metrics']['all']
    print(f"Overall MAE: {am.get('mae_pooled', 0):.2f}")
    print(f"Overall RMSE: {am.get('rmse_pooled', 0):.2f}")
    print(f"Overall Pearson: {am.get('pearson_mean', 0):.3f}")
    
    # building metrics were extracted differently, but they are there
    bm = r['agg_metrics'].get('building')
    if bm:
        print(f"Building MAE: {bm.get('mae_pooled', 0):.2f}")
        print(f"Building RMSE: {bm.get('rmse_pooled', 0):.2f}")
        print(f"Building Bias: {bm.get('bias_pooled', 0):.2f}")
    
    # extra metrics
    em = r['extra_metrics']
    print(f"Predicted Max: {em.get('predicted_max', 0):.2f}")
    print(f"Predicted P99: {em.get('predicted_p99', 0):.2f}")
    print(f"Predicted P95: {em.get('predicted_p95', 0):.2f}")
    
    binned = { b['lo']: b for b in r['binned'] }
    for lo in [0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0]:
        b = binned.get(lo)
        if b:
            print(f"{b['lo']}-{b['hi']}m: MAE={b.get('mae',0):.2f}, Bias={b.get('bias',0):.2f}, Mean Pred={b.get('mean_pred',0):.2f}")
