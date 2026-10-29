"""Test topological spatial smoothing on speed residuals along freeway mainline order.
Evaluates exact RMSE on held-out February validation truth across corridors.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path("trafficflowbench").resolve()
DATA = ROOT / "data" / "kaggle_public"
ASSETS = ROOT / "kaggle_research" / "anubis_assets"
MODELS_DIR = ROOT / "outputs" / "models"

os.environ["TFB_DATA_ROOT"] = str(DATA)
os.environ["TFB_OUTPUT_ROOT"] = str(ROOT / "scratch")
os.environ["TFB_THREADS"] = "6"
os.environ["TFB_PREDICT_DEVICE"] = "CPU"

sys.path.insert(0, str(ASSETS))
import state_model as base
import prepare_state


def main():
    t0 = time.time()
    print("=" * 75)
    print(">>> TESTING TOPOLOGICAL SPATIAL SMOOTHING ON VALIDATION SPEED <<<")
    print("=" * 75)

    # Load speed models
    print("Loading models...")
    cb_d8 = CatBoostRegressor()
    cb_d8.load_model(str(ASSETS / "speed_d8.cbm"))
    cb_d10 = CatBoostRegressor()
    cb_d10.load_model(str(ASSETS / "speed_d10.cbm"))
    m1 = lgb.Booster(model_file=str(MODELS_DIR / "lgb_speed_residual_deep127.txt"))
    m2 = lgb.Booster(model_file=str(MODELS_DIR / "lgb_speed_residual_deep127_seed42.txt"))

    # S24 weights
    w_cb = 0.57556
    w_m1 = 0.06731
    w_m2 = 0.35713

    # Test corridors
    test_panels = ["D7_I10_E", "D7_I210_E", "D7_I405_N", "D12_I5_N"]
    
    # Store per-corridor matrices
    corridor_evals = []

    for pi, name in enumerate(test_panels):
        p = base.Panel(name, pi)
        # Load topology to get link ordering
        topo_file = DATA / "corridors" / name / "network" / "lwr_mainline_topology.csv"
        topo = pd.read_csv(topo_file)
        topo["mainline_link_id"] = topo.mainline_link_id.astype(str)
        # Sort links by order_index
        topo_sorted = topo.sort_values("order_index")
        ordered_links = [l for l in topo_sorted["mainline_link_id"] if l in p.links]
        link_to_order = {l: i for i, l in enumerate(ordered_links)}
        
        times, v, q, e = base.matrices(base.month_frame(name, "2031-02"), p)
        mask, regime = base.simulated_mask(name, times, v, q, e)
        rows, cols = base.sampled_targets(mask, times, 1500)
        
        X, bv, _ = prepare_state.features(
            p, pi, times, np.where(mask, np.nan, v), np.where(mask, np.nan, q), e, rows, cols, regime
        )
        
        # Predict ensemble residual
        p_cb = 0.5 * cb_d8.predict(X, thread_count=6) + 0.5 * cb_d10.predict(X, thread_count=6)
        p_m1 = m1.predict(X, num_threads=6)
        p_m2 = m2.predict(X, num_threads=6)
        res_v = w_cb * p_cb + w_m1 * p_m1 + w_m2 * p_m2
        
        v_pred_raw = bv + res_v
        v_true = v[rows, cols]
        
        valid = np.isfinite(v_true) & (v_true > 1.0) & np.isfinite(v_pred_raw)
        
        corridor_evals.append({
            "panel": name,
            "ordered_links": ordered_links,
            "link_to_order": link_to_order,
            "p_links": p.links,
            "rows": rows[valid],
            "cols": cols[valid],
            "v_raw": v_pred_raw[valid],
            "v_true": v_true[valid],
        })
        print(f"Loaded {name}: {valid.sum():,} validation cells")

    # Evaluate baseline RMSE
    all_raw = np.concatenate([c["v_raw"] for c in corridor_evals])
    all_true = np.concatenate([c["v_true"] for c in corridor_evals])
    base_rmse = float(np.sqrt(np.mean((all_raw - all_true) ** 2)))
    print(f"\nBaseline Validation Speed RMSE: {base_rmse:.4f} km/h")

    # Grid search spatial smoothing factor lambda
    print("\n--- Testing 3-Tap Spatial Smoothing Along Mainline ---")
    for lam in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]:
        smoothed_list = []
        for c in corridor_evals:
            # Build 2D grid for time x order_link
            n_order = len(c["ordered_links"])
            v_grid = {}
            for r, col, v_val in zip(c["rows"], c["cols"], c["v_raw"]):
                lid = c["p_links"][col]
                if lid in c["link_to_order"]:
                    oi = c["link_to_order"][lid]
                    v_grid[(r, oi)] = v_val

            # Apply smoothing: v_smooth(t, oi) = (1-lam)*v(t, oi) + (lam/2)*(v(t, oi-1) + v(t, oi+1))
            v_smooth = []
            for r, col, v_val in zip(c["rows"], c["cols"], c["v_raw"]):
                lid = c["p_links"][col]
                if lid in c["link_to_order"]:
                    oi = c["link_to_order"][lid]
                    left = v_grid.get((r, oi - 1), v_val)
                    right = v_grid.get((r, oi + 1), v_val)
                    vs = (1.0 - lam) * v_val + 0.5 * lam * (left + right)
                    v_smooth.append(vs)
                else:
                    v_smooth.append(v_val)
            smoothed_list.append(np.array(v_smooth))

        all_sm = np.concatenate(smoothed_list)
        sm_rmse = float(np.sqrt(np.mean((all_sm - all_true) ** 2)))
        gain = sm_rmse - base_rmse
        marker = " *** BEST ***" if gain < 0 and sm_rmse == min(sm_rmse, base_rmse) else ""
        print(f"  lambda = {lam:.2f} -> Speed RMSE: {sm_rmse:.4f} km/h (Delta = {gain:+.5f} km/h){marker}")

    print("=" * 75)


if __name__ == "__main__":
    main()
