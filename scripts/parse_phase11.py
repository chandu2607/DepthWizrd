import json
with open('runs/phase11_input_ablation/results.json') as f:
    res = json.load(f)
for r in res['runs']:
    if r['mode'] == 'depth':
        print('Seed', r['seed'])
        agg = r['test']['aggregate']
        print(f"Overall MAE: {agg['all'].get('mae_pooled', 0):.2f}")
        binned = agg.get('binned_all', [])
        for b in binned:
            if b['lo'] >= 30:
                print(f"{b['lo']}-{b['hi']}m: Bias={b.get('bias', 0):.2f}, TrueMean={b.get('mean_gt', 0):.2f}, PredMean={b.get('mean_pred', 0):.2f}")
        print()
