import json

with open('runs/phase14d_correct_decoder_adapt/results.json', 'r') as f:
    res = json.load(f)

for s in res:
    seed = s['seed']
    print(f'=== SEED {seed} ===')
    agg = s['agg_metrics']['all']
    print(f'MAE: {agg.get("mae_pooled", 0):.2f}')
    print(f'RMSE: {agg.get("rmse_pooled", 0):.2f}')
    print(f'Pearson: {agg.get("pearson_mean", 0):.3f}')
    print(f'Best Epoch: {s["best_epoch"]}')
    
    binned = s['binned']
    for b in binned:
        if b['lo'] >= 30:
            print(f"{b['lo']}-{b['hi']}m: Bias={b['bias']:.2f}, Mean GT={b['mean_gt']:.2f}, Mean Pred={b['mean_pred']:.2f}")
    
    # We also want to find if there was a crazy outlier that caused the massive RMSE in seed 0.
    # The scene metrics were not saved in results.json unfortunately, only the aggregated metrics.
    # But wait, in the binned metrics, if there is a massive spike, we might see it in Mean Pred or RMSE of a specific bin.
    for b in binned:
        print(f"  Bin {b['lo']}-{b['hi']}m: RMSE={b.get('rmse', 0):.2f}")
