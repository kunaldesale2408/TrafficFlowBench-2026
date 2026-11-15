import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("trafficflowbench")
DATA = ROOT / "data" / "kaggle_public"

corridors = sorted([p.name for p in (DATA / "corridors").iterdir() if p.is_dir()])

train_meds = {}
val_meds = {}

for p in corridors:
    # Train
    unm_train = sorted((DATA / "corridors" / p / "train" / "mainline_states").glob("**/*.parquet"))
    dfs_tr = [pd.read_parquet(f, columns=['link_id', 'speed_kmh']) for f in unm_train[-60:]]
    df_tr = pd.concat(dfs_tr, ignore_index=True)
    m_tr = df_tr[df_tr['speed_kmh'] >= 90.0].groupby('link_id')['speed_kmh'].median().to_dict()
    for l, v in m_tr.items():
        train_meds[(p, str(l))] = float(v)
        
    # Validation observed cells
    msk_val = sorted((DATA / "corridors" / p / "validation" / "mainline_states_masked").glob("**/*.parquet"))
    dfs_val = [pd.read_parquet(f, columns=['link_id', 'speed_kmh']) for f in msk_val]
    df_val = pd.concat(dfs_val, ignore_index=True).dropna(subset=['speed_kmh'])
    m_val = df_val[df_val['speed_kmh'] >= 90.0].groupby('link_id')['speed_kmh'].median().to_dict()
    for l, v in m_val.items():
        val_meds[(p, str(l))] = float(v)

common_keys = sorted(set(train_meds.keys()) & set(val_meds.keys()))
print(f"Common corridor-link pairs: {len(common_keys):,}")

tr_arr = np.array([train_meds[k] for k in common_keys])
val_arr = np.array([val_meds[k] for k in common_keys])

diff = val_arr - tr_arr
print(f"Correlation between Train & Val link free-flow medians: {np.corrcoef(tr_arr, val_arr)[0, 1]:.5f}")
print(f"Mean difference: {diff.mean():.4f} km/h")
print(f"MAE difference : {np.abs(diff).mean():.4f} km/h")
print(f"Max difference : {np.abs(diff).max():.4f} km/h")
print(f"Quantiles of |diff|:")
print(pd.Series(np.abs(diff)).describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))
