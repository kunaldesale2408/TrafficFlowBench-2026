import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("trafficflowbench")
s44 = pd.read_csv(ROOT / "outputs" / "submissions" / "queue_s44_precision.csv")
probs = pd.read_parquet(ROOT / "scratch" / "test_ongoing_probabilities.parquet")

wi_files = list(Path('trafficflowbench/data/kaggle_public/task2').glob('*/*/window_index.csv'))
wi_df = pd.concat([pd.read_csv(f) for f in wi_files], ignore_index=True)
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

d7_405 = m[(m['split'] == 'validation') & (m['condition'] == 'queue_ongoing') & (m['panel'] == 'D7_I405_N')]
print("D7_I405_N validation ongoing windows:", d7_405['window_id'].unique())
for wid, grp in d7_405.groupby('window_id'):
    print(f"\nWindow {wid}:")
    pos = grp[grp['queue_pred'] == 1]
    print(f"  Total positives: {len(pos)}")
    print(f"  Positives with p_s24 < 0.20: {(pos['p_s24'] < 0.20).sum()}")
    print(f"  Positives with p_s24 < 0.30: {(pos['p_s24'] < 0.30).sum()}")
    print(f"  Positives with p_s24 < 0.40: {(pos['p_s24'] < 0.40).sum()}")
    print(f"  p_s24 summary for positives: min={pos['p_s24'].min():.3f}, mean={pos['p_s24'].mean():.3f}, max={pos['p_s24'].max():.3f}")
