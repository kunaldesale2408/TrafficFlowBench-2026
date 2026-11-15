import pandas as pd
import numpy as np
from pathlib import Path
import json

ROOT = Path("trafficflowbench")
DATA = ROOT / "data" / "kaggle_public"

# Let's test on D12_I5_N across 20 held-out masked days
p = "D12_I5_N"
panel_dir = DATA / "corridors" / p
unm_files = sorted((panel_dir / "train" / "mainline_states").glob("**/*.parquet"))
msk_files = sorted((panel_dir / "train" / "mainline_states_masked").glob("**/*.parquet"))

print(f"Total unmasked files: {len(unm_files)}, masked files: {len(msk_files)}")

# 1. Compute empirical link free-flow median from the first 200 days
dfs = []
for f in unm_files[:200]:
    dfs.append(pd.read_parquet(f, columns=['link_id', 'speed_kmh']))
df_train = pd.concat(dfs, ignore_index=True)
ff_train = df_train[df_train['speed_kmh'] >= 90.0]
link_ff_map = ff_train.groupby('link_id')['speed_kmh'].median().to_dict()
print(f"Computed link FF medians for {len(link_ff_map)} links.")

# 2. Test on the last 20 masked days (which have corresponding unmasked truth!)
# In these masked days, let's compare:
# - A simple linear interpolator / baseline model vs shrunk baseline
test_unm = unm_files[-20:]
test_msk = msk_files[-20:]

true_speeds = []
base_speeds = []
link_ids = []

for u_f, m_f in zip(test_unm, test_msk):
    u_df = pd.read_parquet(u_f)
    m_df = pd.read_parquet(m_f)
    
    # Masked target cells
    mask = m_df['speed_kmh'].isna() & u_df['is_score_eligible'].astype(bool)
    if not mask.any(): continue
    
    # Target speeds
    y_true = u_df.loc[mask, 'speed_kmh'].values
    l_ids = u_df.loc[mask, 'link_id'].astype(str).values
    
    # Let's simulate a standard high-quality baseline (e.g. forward-backward fill + noise or link profile)
    # Even better: let's use the actual day-level median profile
    day_prof = m_df.groupby('link_id')['speed_kmh'].median().to_dict()
    y_base = np.array([day_prof.get(l, link_ff_map.get(l, 115.0)) for l in l_ids])
    
    true_speeds.extend(y_true)
    base_speeds.extend(y_base)
    link_ids.extend(l_ids)

y_true = np.array(true_speeds)
y_base = np.array(base_speeds)
l_arr = np.array(link_ids)

# Filter to free flow ground truth
ff_mask = y_true >= 90.0
print(f"\nTotal evaluated cells: {len(y_true):,}, Free flow cells: {ff_mask.sum():,} ({ff_mask.mean():.2%})")

base_rmse = np.sqrt(((y_base[ff_mask] - y_true[ff_mask])**2).mean())
print(f"Baseline Free Flow RMSE: {base_rmse:.4f} km/h")

# Test shrinkage towards global link FF median
link_ff_arr = np.array([link_ff_map.get(l, 115.0) for l in l_arr])
for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
    y_shrunk = (1 - alpha) * y_base + alpha * link_ff_arr
    rmse = np.sqrt(((y_shrunk[ff_mask] - y_true[ff_mask])**2).mean())
    print(f"  alpha = {alpha:.1f} -> RMSE = {rmse:.4f} km/h (Delta: {rmse - base_rmse:+.4f})")
