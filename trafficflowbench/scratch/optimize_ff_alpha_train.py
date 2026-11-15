import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("trafficflowbench")
DATA = ROOT / "data" / "kaggle_public"

corridors = sorted([p.name for p in (DATA / "corridors").iterdir() if p.is_dir()])

# Compute empirical free-flow speed medians from train
link_ff = {}
for p in corridors:
    unm_files = sorted((DATA / "corridors" / p / "train" / "mainline_states").glob("**/*.parquet"))
    dfs = [pd.read_parquet(f, columns=['link_id', 'speed_kmh']) for f in unm_files[:200]]
    df_p = pd.concat(dfs, ignore_index=True)
    meds = df_p[df_p['speed_kmh'] >= 90.0].groupby('link_id')['speed_kmh'].median().to_dict()
    for l, val in meds.items():
        link_ff[(p, str(l))] = float(val)

# Evaluate on 10 held-out days across ALL 10 corridors
all_true = []
all_base = []
all_lff = []

for p in corridors:
    unm_files = sorted((DATA / "corridors" / p / "train" / "mainline_states").glob("**/*.parquet"))
    msk_files = sorted((DATA / "corridors" / p / "train" / "mainline_states_masked").glob("**/*.parquet"))
    
    for u_f, m_f in zip(unm_files[-10:], msk_files[-10:]):
        u_df = pd.read_parquet(u_f)
        m_df = pd.read_parquet(m_f)
        
        mask = m_df['speed_kmh'].isna() & u_df['is_score_eligible'].astype(bool)
        if not mask.any(): continue
        
        y_true = u_df.loc[mask, 'speed_kmh'].to_numpy(dtype=float)
        l_ids = u_df.loc[mask, 'link_id'].astype(str).values
        
        # Spatial-temporal day interpolation as baseline
        m_pivot = m_df.pivot(index='timestamp', columns='link_id', values='speed_kmh')
        m_interp = m_pivot.interpolate(method='linear', limit_direction='both').fillna(115.0)
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

base_rmse = np.sqrt(((y_base - y_true)**2).mean())
print(f"Baseline Overall RMSE: {base_rmse:.4f} km/h on {len(y_true):,} cells")

results = []
for tau in [98.0, 100.0, 102.0, 104.0, 105.0]:
    for alpha in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        gate = y_base >= tau
        y_cand = y_base.copy()
        y_cand[gate] = (1 - alpha) * y_base[gate] + alpha * y_lff[gate]
        y_cand = np.clip(y_cand, 6.0, 120.75)
        rmse = np.sqrt(((y_cand - y_true)**2).mean())
        results.append({"tau": tau, "alpha": alpha, "rmse": rmse, "delta": rmse - base_rmse})

res_df = pd.DataFrame(results).sort_values("rmse")
print("\nTop 15 Parameter Combinations by Overall RMSE:")
print(res_df.head(15).to_string(index=False))
