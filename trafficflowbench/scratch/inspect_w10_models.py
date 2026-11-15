import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("trafficflowbench")
s44 = pd.read_csv(ROOT / "outputs" / "submissions" / "queue_s44_precision.csv")
probs = pd.read_parquet(ROOT / "scratch" / "test_ongoing_probabilities.parquet")

w10 = s44[s44['window_id'] == 'Q2_D7_I405_N_validation_010'].copy()

orig = "2031-03-28T09:00:00Z" # let's get origin from window_index
wi = pd.read_csv(ROOT / "data" / "kaggle_public" / "task2" / "D7_I405_N" / "validation" / "window_index.csv")
orig = wi.loc[wi['window_id'] == 'Q2_D7_I405_N_validation_010', 'forecast_origin'].iloc[0]
w10['h'] = ((pd.to_datetime(w10['timestamp'], utc=True) - pd.to_datetime(orig, utc=True)).dt.total_seconds() // 300).astype(int)

p_keys = ['window_id', 'h', 'link_id']
for k in p_keys:
    probs[k] = probs[k].astype(str)
    w10[k] = w10[k].astype(str)

m = w10.merge(probs[p_keys + ['p_s24', 'p_lgb', 'p_cb']], on=p_keys, how='inner')

print("Window 010 predictions vs p_s24:")
for h, grp in m.groupby('h'):
    pos = grp[grp['queue_pred'] == 1]
    print(f"h={h}: {len(pos)} positives / {len(grp)} links. p_s24 for pos: min={pos['p_s24'].min():.3f}, median={pos['p_s24'].median():.3f}")

# Also check Semwal's raw probability if available
semwal_sub = pd.read_csv(ROOT / "kaggle_research" / "semwal_087" / "output" / "submission.csv")
semwal_q = semwal_sub[semwal_sub['task'] == 'queue'].copy()
key = pd.read_csv(ROOT / "data" / "kaggle_public" / "submission_key.csv")
semwal_q['window_id'] = key.loc[key['task'] == 'queue', 'window_id'].values
semwal_w10 = semwal_q[semwal_q['window_id'] == 'Q2_D7_I405_N_validation_010']
print(f"\nSemwal 087 positives on window 010: {semwal_w10['queue_pred'].sum()}")
