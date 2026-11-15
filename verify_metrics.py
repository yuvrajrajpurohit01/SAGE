import json
import numpy as np

with open('runs/journal-eval/results.json') as f:
    data = json.load(f)

SEEDS = ['seed42','seed43','seed44','seed45','seed46']

agreement = []
false_esc = []
missed_esc = []
regret = []
heavy_call_rate = []
d1_rejection = []
sage_rmse = []
oracle_rmse = []
small_rmse = []
heavy_rmse = []
fusion_rmse = []
qs_means = []

for seed in SEEDS:
    sd = data['per_seed'][seed]
    small = sd['small_only']
    heavy = sd['numerical_heavy']
    sage = sd['numerical_sage']
    fusion = sd['always_average']

    truth = np.array(small['truth'])
    small_pred = np.array(small['predictions'])
    heavy_pred = np.array(heavy['predictions'])
    sage_pred = np.array(sage['predictions'])
    fusion_pred = np.array(fusion['predictions'])
    sage_sources = np.array(sage['sources'])

    small_ae = np.abs(small_pred - truth)
    heavy_ae = np.abs(heavy_pred - truth)
    oracle_better_heavy = heavy_ae < small_ae
    oracle_ae = np.minimum(small_ae, heavy_ae)

    sage_picked_heavy = sage_sources == 'heavy'
    sage_picked_small = sage_sources == 'small'
    sage_picked_fusion = sage_sources == 'fusion'

    agree_heavy = sage_picked_heavy & oracle_better_heavy
    agree_small = sage_picked_small & (~oracle_better_heavy)
    agreement.append((agree_heavy | agree_small).mean())

    false_esc.append((sage_picked_heavy & (~oracle_better_heavy)).mean())
    missed_esc.append((sage_picked_small & oracle_better_heavy).mean())

    sage_ae = np.abs(sage_pred - truth)
    regret.append((sage_ae - oracle_ae).mean())

    heavy_call_rate.append(sage_picked_heavy.mean())
    d1_rejection.append((sage_picked_heavy | sage_picked_fusion).mean())

    sage_rmse.append(np.sqrt(np.mean(sage_ae**2)))
    oracle_rmse.append(np.sqrt(np.mean(oracle_ae**2)))
    small_rmse.append(np.sqrt(np.mean(small_ae**2)))
    heavy_rmse.append(np.sqrt(np.mean(((heavy_pred-truth)**2))))
    fusion_rmse.append(fusion['rmse'])
    qs_means.append(sage['mean_q_s'])

def pop_stats(vals):
    m = np.mean(vals)
    s = np.std(vals)
    lo = np.percentile(vals, 2.5)
    hi = np.percentile(vals, 97.5)
    return m, s, lo, hi

print('=== AGGREGATE (population std, n=5 seeds) ===')
for name, vals in [('Agreement', agreement), ('False Esc', false_esc),
                   ('Missed Esc', missed_esc), ('Regret', regret),
                   ('Heavy Call Rate', heavy_call_rate),
                   ('D1 Rejection (Esc)', d1_rejection)]:
    m, s, lo, hi = pop_stats(vals)
    print(f'{name}: mean={m:.4f}, pop_std={s:.4f}, 95% CI=[{lo:.4f}, {hi:.4f}]')

print()
print('=== RMSE (mean ± pop_std) ===')
for name, vals in [('SAGE', sage_rmse), ('Oracle', oracle_rmse), ('Small', small_rmse),
                   ('Heavy', heavy_rmse), ('Fusion', fusion_rmse)]:
    m, s, lo, hi = pop_stats(vals)
    print(f'{name}: {m:.4f} ± {s:.4f}, 95% CI=[{lo:.4f}, {hi:.4f}]')

print()
print('Qs means per seed:')
for seed, q in zip(SEEDS, qs_means):
    print(f'  {seed}: mean_q_s={q:.4f}')

print()
print('Per-seed D1 rejection (escalation = heavy+fusion):')
for seed, dr in zip(SEEDS, d1_rejection):
    print(f'  {seed}: {dr:.3f} ({int(dr*60)}/60 engines)')

print()
print('Per-seed heavy-call rate (source==heavy only):')
for seed, hcr in zip(SEEDS, heavy_call_rate):
    print(f'  {seed}: {hcr:.3f} ({int(hcr*60)}/60 engines)')
