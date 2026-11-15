import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("trafficflowbench")
DATA = ROOT / "data" / "kaggle_public"

corridors = sorted([p.name for p in (DATA / "corridors").iterdir() if p.is_dir()])

# Compute empirical free-flow speed medians per corridor and link from train months (first 200 days)
print("Computing empirical link free-flow medians from first 200 days...")
link_ff = {}
for p in corridors:
    unm_files = sorted((DATA / "corridors" / p / "train" / "mainline_states").glob("**/*.parquet"))
    dfs = []
    for f in unm_files[:200]:
        dfs.append(pd.read_parquet(f, columns=['link_id', 'speed_kmh']))
    df_p = pd.concat(dfs, ignore_index=True)
    ff = df_p[df_p['speed_kmh'] >= 90.0]
    meds = ff.groupby('link_id')['speed_kmh'].median().to_dict()
    for l, val in meds.items():
        link_ff[(p, str(l))] = float(val)

print(f"Computed medians for {len(link_ff)} corridor-link pairs.")

# Now test on 5 held-out days across ALL 10 corridors
all_true = []
all_base = []
all_lff = []
all_is_cong = []

for p in corridors:
    unm_files = sorted((DATA / "corridors" / p / "train" / "mainline_states").glob("**/*.parquet"))
    msk_files = sorted((DATA / "corridors" / p / "train" / "mainline_states_masked").glob("**/*.parquet"))
    
    test_u = unm_files[-5:]
    test_m = msk_files[-5:]
    
    for u_f, m_f in zip(test_u, test_m):
        u_df = pd.read_parquet(u_f)
        m_df = pd.read_parquet(m_f)
        
        mask = m_df['speed_kmh'].isna() & u_df['is_score_eligible'].astype(bool)
        if not mask.any(): continue
        
        y_true = u_df.loc[mask, 'speed_kmh'].to_numpy(dtype=float)
        l_ids = u_df.loc[mask, 'link_id'].astype(str).values
        
        # Simple spatial-temporal baseline (nearest observed cell in day)
        # To simulate high-quality model, let's interpolate inside day
        m_pivot = m_df.pivot(index='timestamp', columns='link_id', values='speed_kmh')
        m_interp = m_pivot.interpolate(method='linear', limit_direction='both').fillna(115.0)
        
        # Map interpolated predictions
        pred_dict = m_interp.stack().to_dict()
        ts_arr = u_df.loc[mask, 'timestamp'].values
        y_pred = np.array([pred_dict.get((t, l), link_ff.get((p, l), 115.0)) for t, l in zip(ts_arr, l_ids)])
        
        l_ff_arr = np.array([link_ff.get((p, l), 115.0) for l in l_ids])
        
        all_true.extend(y_true)
        all_base.extend(y_pred)
        all_lff.extend(l_ff_arr)

y_true = np.array(all_true)
y_base = np.array(all_base)
y_lff = np.array(all_lff)

print(f"\nTotal evaluated cells across 10 corridors: {len(y_true):,}")
base_overall_rmse = np.sqrt(((y_base - y_true)**2).mean())
print(f"Overall Baseline Speed RMSE: {base_overall_rmse:.4f} km/h")

# Test gating: only apply shrinkage if y_base >= tau_free
print("\n" + "="*65)
print(f"{'tau_free':<10} | {'alpha':<8} | {'Overall RMSE':<14} | {'Delta':<10} | {'Cells Shrunk':<12}")
print("-" * 65)

for tau in [95.0, 100.0, 105.0, 108.0, 110.0]:
    for alpha in [0.3, 0.5, 0.7, 0.8, 0.9, 1.0]:
        gate = y_base >= tau
        y_cand = y_base.copy()
        y_cand[gate] = (1 - alpha) * y_base[gate] + alpha * y_lff[gate]
        # Physical speed cap
        y_cand = np.clip(y_cand, 6.0, 120.75)
        rmse = np.sqrt(((y_cand - y_true)**2).mean())
        delta = rmse - base_overall_rmse
        print(f"{tau:<10.1f} | {alpha:<8.2f} | {rmse:<14.4f} | {delta:<+10.4f} | {gate.sum():<12,}")
