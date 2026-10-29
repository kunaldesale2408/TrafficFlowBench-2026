"""Optimize simplex weights for Task 1 flow models:
  - CatBoost (bundled d8/d10)
  - Deep-LGB127 flow (lgb_flow_residual_deep127.txt)
  - 94f-LGB flow (lgb_flow_residual_94f.txt)
Evaluates flow RMSE on the official validation month (2031-02).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize

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
from state_inference import StatePredictor


def main():
    t0 = time.time()
    print("=" * 75)
    print(">>> OPTIMIZING MULTI-MODEL TASK 1 FLOW ENSEMBLE WEIGHTS <<<")
    print("=" * 75)

    # 1. Collect validation data (February 2031-02 across all 10 panels)
    val_x, val_yq, val_bq, val_lanes = [], [], [], []
    panels = base.all_panels()
    val_month = base.VALID_MONTH

    print(f"Collecting validation targets from month {val_month} across {len(panels)} panels...")
    for pi, name in enumerate(panels):
        t_p = time.time()
        p = base.Panel(name, pi)
        times, v, q, e = base.matrices(base.month_frame(name, val_month), p)
        mask, regime = base.simulated_mask(name, times, v, q, e)
        rows, cols = base.sampled_targets(mask, times, 2000)
        if len(rows) > 0:
            X, _, bq = prepare_state.features(
                p, pi, times, np.where(mask, np.nan, v), np.where(mask, np.nan, q), e, rows, cols, regime
            )
            rq = q[rows, cols] / p.lanes[cols] - bq
            valid_idx = np.isfinite(rq)
            val_x.append(X[valid_idx])
            val_yq.append(rq[valid_idx])
            val_bq.append(bq[valid_idx])
            val_lanes.append(p.lanes[cols[valid_idx]])
        print(f"  {name}: {valid_idx.sum():,} validation samples ({time.time()-t_p:.1f}s)")

    X_val = np.vstack(val_x)
    yq_val = np.concatenate(val_yq)
    print(f"\nTotal Validation Samples: {len(yq_val):,}, Feature Dimension: {X_val.shape[1]}")

    # 2. Load models
    print("\nLoading CatBoost benchmark...")
    cat_predictor = StatePredictor(ASSETS)
    pred_cat = cat_predictor.residual(X_val, "flow")

    print("Loading Deep LightGBM flow (d127)...")
    m_deep = lgb.Booster(model_file=str(MODELS_DIR / "lgb_flow_residual_deep127.txt"))
    pred_deep = m_deep.predict(X_val, num_threads=6)

    print("Loading 94f LightGBM flow...")
    m_94f = lgb.Booster(model_file=str(MODELS_DIR / "lgb_flow_residual_94f.txt"))
    pred_94f = m_94f.predict(X_val, num_threads=6)

    # 3. Individual model performance
    rmse_cat = float(np.sqrt(np.mean((yq_val - pred_cat) ** 2)))
    rmse_deep = float(np.sqrt(np.mean((yq_val - pred_deep) ** 2)))
    rmse_94f = float(np.sqrt(np.mean((yq_val - pred_94f) ** 2)))
    rmse_s24 = float(np.sqrt(np.mean((yq_val - (0.95 * pred_cat + 0.05 * pred_deep)) ** 2)))

    print("-" * 75)
    print(f"Individual Model Validation Flow Residual RMSE (per lane):")
    print(f"  - CatBoost (bundled d8+d10): {rmse_cat:.4f} vph")
    print(f"  - Deep-LGB127:               {rmse_deep:.4f} vph")
    print(f"  - 94f-LGB:                   {rmse_94f:.4f} vph")
    print(f"  - S24 blend (95% CB / 5% DL): {rmse_s24:.4f} vph")
    print("-" * 75)

    # 4. Continuous SLSQP Simplex Optimization
    def loss_fn(w):
        pred = w[0] * pred_cat + w[1] * pred_deep + w[2] * pred_94f
        return float(np.sqrt(np.mean((yq_val - pred) ** 2)))

    bnds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    init_w = [0.80, 0.10, 0.10]

    opt = minimize(loss_fn, init_w, method="SLSQP", bounds=bnds, constraints=cons)
    w_best = opt.x
    rmse_opt = opt.fun

    print("\nOPTIMAL CONTINUOUS CONVEX ENSEMBLE WEIGHTS:")
    print(f"  - CatBoost (d8+d10):         {w_best[0] * 100:.2f}%")
    print(f"  - Deep-LGB127:               {w_best[1] * 100:.2f}%")
    print(f"  - 94f-LGB:                   {w_best[2] * 100:.2f}%")
    print(f"  - Optimal Flow RMSE:         {rmse_opt:.4f} vph (Gain over S24: {rmse_opt - rmse_s24:+.4f})")
    print("=" * 75)

    # Save to outputs/models
    out_cfg = {
        "rmse_cat": rmse_cat,
        "rmse_deep": rmse_deep,
        "rmse_94f": rmse_94f,
        "rmse_s24": rmse_s24,
        "rmse_optimal": rmse_opt,
        "gain": rmse_opt - rmse_s24,
        "weights": {
            "catboost": float(w_best[0]),
            "lgb_deep127": float(w_best[1]),
            "lgb_94f": float(w_best[2]),
        },
    }
    cfg_path = MODELS_DIR / "flow_ensemble_weights_s26.json"
    with open(cfg_path, "w") as f:
        json.dump(out_cfg, f, indent=2)
    print(f"Saved optimal flow blend configuration -> {cfg_path.name}")


if __name__ == "__main__":
    main()
