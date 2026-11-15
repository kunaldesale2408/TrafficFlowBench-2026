import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor
import os
import sys

ROOT = Path("trafficflowbench")
DATA = ROOT / "data" / "kaggle_public"
ASSETS = ROOT / "kaggle_research" / "anubis_assets"

sys.path.insert(0, str(ASSETS))
from state_model import StateBaseline, NetworkTopology, StationMetadata
from prepare_state import ContextFeatures

# Load CatBoost models
print("Loading CatBoost speed models...")
cb_d8 = CatBoostRegressor()
cb_d8.load_model(str(ASSETS / "speed_d8.cbm"))
cb_d10 = CatBoostRegressor()
cb_d10.load_model(str(ASSETS / "speed_d10.cbm"))

# Let's test on D12_I5_N, 5 held-out train days
p = "D12_I5_N"
panel_dir = DATA / "corridors" / p
unm_files = sorted((panel_dir / "train" / "mainline_states").glob("**/*.parquet"))
msk_files = sorted((panel_dir / "train" / "mainline_states_masked").glob("**/*.parquet"))

# Compute link free-flow medians from first 200 days
print("Computing train link FF medians...")
dfs = [pd.read_parquet(f, columns=['link_id', 'speed_kmh']) for f in unm_files[:200]]
df_tr = pd.concat(dfs, ignore_index=True)
link_ff = df_tr[df_tr['speed_kmh'] >= 90.0].groupby('link_id')['speed_kmh'].median().to_dict()

# Evaluate on 5 test days
y_true_list = []
y_pred_list = []
l_ids_list = []

for u_f, m_f in zip(unm_files[-5:], msk_files[-5:]):
    u_df = pd.read_parquet(u_f)
    m_df = pd.read_parquet(m_f)
    
    mask = m_df['speed_kmh'].isna() & u_df['is_score_eligible'].astype(bool)
    if not mask.any(): continue
    
    # Truth
    y_true = u_df.loc[mask, 'speed_kmh'].to_numpy(dtype=float)
    l_ids = u_df.loc[mask, 'link_id'].astype(str).values
    
    # Run CatBoost inference
    # Load day context
    base = StateBaseline(p, DATA / "corridors" / p)
    # Get feature matrix
    X_d = base.prepare_features(m_df)
    # Predict
    pred_d8 = cb_d8.predict(X_d)
    pred_d10 = cb_d10.predict(X_d)
    pred_cb = 0.5 * pred_d8 + 0.5 * pred_d10
    
    # Subselect masked rows
    y_pred = pred_cb[mask]
    
    y_true_list.extend(y_true)
    y_pred_list.extend(y_pred)
    l_ids_list.extend(l_ids)

y_true = np.array(y_true_list)
y_pred = np.array(y_pred_list)
l_arr = np.array(l_ids_list)
y_lff = np.array([link_ff.get(l, 115.0) for l in l_arr])

print(f"\nEvaluated {len(y_true):,} masked cells on {p}.")
base_rmse = np.sqrt(((y_pred - y_true)**2).mean())
print(f"CatBoost Baseline Speed RMSE: {base_rmse:.4f} km/h")

# Grid search shrinkage
print("\n" + "="*60)
print(f"{'tau_free':<10} | {'alpha':<8} | {'RMSE (km/h)':<14} | {'Delta':<10}")
print("-" * 60)

for tau in [90.0, 95.0, 100.0, 105.0, 108.0]:
    for alpha in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        gate = y_pred >= tau
        y_cand = y_pred.copy()
        y_cand[gate] = (1 - alpha) * y_pred[gate] + alpha * y_lff[gate]
        y_cand = np.clip(y_cand, 6.0, 120.75)
        rmse = np.sqrt(((y_cand - y_true)**2).mean())
        print(f"{tau:<10.1f} | {alpha:<8.2f} | {rmse:<14.4f} | {rmse - base_rmse:+.4f}")
