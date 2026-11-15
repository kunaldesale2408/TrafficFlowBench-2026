import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("trafficflowbench")
DATA = ROOT / "data" / "kaggle_public"

s44 = pd.read_csv(ROOT / "outputs" / "submissions" / "queue_s44_precision.csv")
probs = pd.read_parquet(ROOT / "scratch" / "test_ongoing_probabilities.parquet")

wi_df = pd.concat([pd.read_csv(f) for f in (pd.Series(list(pd.io.common.Path('trafficflowbench/data/kaggle_public/task2').glob('*/*/window_index.csv'))))], ignore_index=True)
cond_map = dict(zip(wi_df['window_id'].astype(str), wi_df['condition'].astype(str)))
orig_map = dict(zip(wi_df['window_id'].astype(str), wi_df['forecast_origin'].astype(str)))
panel_map = dict(zip(wi_df['window_id'].astype(str), wi_df['panel'].astype(str)))

s44['condition'] = s44['window_id'].map(cond_map)
s44['panel'] = s44['window_id'].map(panel_map)
s44['h'] = ((pd.to_datetime(s44['timestamp'], utc=True) - pd.to_datetime(s44['window_id'].map(orig_map), utc=True)).dt.total_seconds() // 300).astype(int)
s44['split'] = np.where(s44['timestamp'].str[:7] == '2031-03', 'validation', 'private')

p_keys = ['window_id', 'h', 'link_id']
for k in p_keys:
    probs[k] = probs[k].astype(str)
    s44[k] = s44[k].astype(str)

m = s44.merge(probs[p_keys + ['p_s24', 'p_lgb', 'p_cb']], on=p_keys, how='inner')

for split in ['validation', 'private']:
    sp_df = m[(m['split'] == split) & (m['condition'] == 'queue_ongoing') & (m['queue_pred'] == 1)]
    print(f"\n{'='*20} {split.upper()} ONGOING QUEUE POSITIVES ({len(sp_df):,} total) {'='*20}")
    print(f"p_s24 distribution: min={sp_df['p_s24'].min():.3f}, mean={sp_df['p_s24'].mean():.3f}, median={sp_df['p_s24'].median():.3f}")
    for thresh in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
        low_count = (sp_df['p_s24'] < thresh).sum()
        print(f"  Positives with p_s24 < {thresh:.2f}: {low_count}")

