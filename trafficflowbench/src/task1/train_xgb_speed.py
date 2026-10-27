"""Train an XGBoost speed residual booster on the 94 spatiotemporal features.
Key characteristics:
  - Architecture: Depth-wise histogram-based trees (tree_method='hist', max_depth=8)
  - Complementary to oblivious trees (CatBoost) and leaf-wise trees (LightGBM)
  - Evaluates 3-way ensembling (CatBoost + LightGBM + XGBoost) on held-out validation
  - Caches feature matrix to scratch/state_cache_1200.npz for instant reuse
Outputs:
  outputs/models/xgb_speed_residual.json
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import xgboost as xgb

ROOT = Path("trafficflowbench").resolve()
DATA = ROOT / "data" / "kaggle_public"
ASSETS = ROOT / "kaggle_research" / "anubis_assets"
OUT_MODELS = ROOT / "outputs" / "models"
SCRATCH = ROOT / "scratch"
OUT_MODELS.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)

CACHE_FILE = SCRATCH / "state_cache_1200.npz"

os.environ["TFB_DATA_ROOT"] = str(DATA)
os.environ["TFB_OUTPUT_ROOT"] = str(SCRATCH)
os.environ["TFB_THREADS"] = "6"
os.environ["TFB_PREDICT_DEVICE"] = "CPU"

sys.path.insert(0, str(ASSETS))
import state_model as base
import prepare_state
from state_inference import StatePredictor


def get_cached_or_collect_dataset(samples_per_day: int = 1200):
    if CACHE_FILE.exists():
        t0 = time.time()
        print(f"Loading cached dataset from {CACHE_FILE.name}...")
        data = np.load(CACHE_FILE)
        X_train = data["X_train"]
        yv_train = data["yv_train"]
        X_val = data["X_val"]
        yv_val = data["yv_val"]
        bv_val = data["bv_val"]
        lanes_val = data["lanes_val"]
        print(f"Cached dataset loaded in {time.time()-t0:.1f}s.")
        print(f"Train set: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
        print(f"Val set:   {X_val.shape[0]:,} samples")
        return (X_train, yv_train), (X_val, yv_val, bv_val, lanes_val)

    t0 = time.time()
    print(f"Collecting training and validation datasets ({samples_per_day} samples/day)...")
    
    train_x, train_yv = [], []
    val_x, val_yv = [], []
    val_bv, val_lanes = [], []
    
    panels = base.all_panels()
    train_months = base.TRAIN_MONTHS
    val_month = base.VALID_MONTH
    
    for pi, name in enumerate(panels):
        p = base.Panel(name, pi)
        # Training months (2030-05 through 2031-01)
        for month in train_months:
            times, v, q, e = base.matrices(base.month_frame(name, month), p)
            mask, regime = base.simulated_mask(name, times, v, q, e)
            rows, cols = base.sampled_targets(mask, times, samples_per_day)
            if len(rows) == 0:
                continue
            X, bv, _ = prepare_state.features(
                p, pi, times, np.where(mask, np.nan, v), np.where(mask, np.nan, q), e, rows, cols, regime
            )
            rv = v[rows, cols] - bv
            valid_idx = np.isfinite(rv)
            train_x.append(X[valid_idx])
            train_yv.append(rv[valid_idx])
        
        # Validation month (February 2031-02)
        times, v, q, e = base.matrices(base.month_frame(name, val_month), p)
        mask, regime = base.simulated_mask(name, times, v, q, e)
        rows, cols = base.sampled_targets(mask, times, 1500)
        if len(rows) > 0:
            X, bv, _ = prepare_state.features(
                p, pi, times, np.where(mask, np.nan, v), np.where(mask, np.nan, q), e, rows, cols, regime
            )
            rv = v[rows, cols] - bv
            valid_idx = np.isfinite(rv)
            val_x.append(X[valid_idx])
            val_yv.append(rv[valid_idx])
            val_bv.append(bv[valid_idx])
            val_lanes.append(p.lanes[cols[valid_idx]])
            
        print(f"  Panel {name}: loaded train and validation")
        
    X_train = np.vstack(train_x)
    yv_train = np.concatenate(train_yv)
    
    X_val = np.vstack(val_x)
    yv_val = np.concatenate(val_yv)
    bv_val = np.concatenate(val_bv)
    lanes_val = np.concatenate(val_lanes)
    
    print(f"Data collection completed in {time.time()-t0:.1f}s.")
    print(f"Train set: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"Val set:   {X_val.shape[0]:,} samples")

    print(f"Saving dataset cache to {CACHE_FILE.name}...")
    np.savez_compressed(
        CACHE_FILE,
        X_train=X_train,
        yv_train=yv_train,
        X_val=X_val,
        yv_val=yv_val,
        bv_val=bv_val,
        lanes_val=lanes_val,
    )
    print(f"Dataset cached successfully ({CACHE_FILE.stat().st_size / (1024*1024):.1f} MB).")
    return (X_train, yv_train), (X_val, yv_val, bv_val, lanes_val)


def main():
    print("=" * 70)
    print(">>> TRAINING XGBOOST SPEED RESIDUAL MODEL & OPTIMIZING 3-WAY BLEND <<<")
    print("=" * 70)
    
    (X_train, yv_train), (X_val, yv_val, bv_val, lanes_val) = get_cached_or_collect_dataset(samples_per_day=1200)
    
    print("\n--- Training XGBoost Speed Residual Booster ---")
    t0 = time.time()
    
    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "max_depth": 8,
        "learning_rate": 0.04,
        "n_estimators": 1000,
        "early_stopping_rounds": 40,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.5,
        "reg_lambda": 3.0,
        "n_jobs": 6,
        "random_state": 20261027,
    }
    
    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(
        X_train,
        yv_train,
        eval_set=[(X_train, yv_train), (X_val, yv_val)],
        verbose=100,
    )
    best_iter = getattr(model_xgb, "best_iteration", model_xgb.n_estimators)
    print(f"XGBoost model trained in {time.time()-t0:.1f}s, best iteration: {best_iter}")
    
    save_path = OUT_MODELS / "xgb_speed_residual.json"
    model_xgb.save_model(str(save_path))
    print(f"Saved XGBoost model -> {save_path.name}")
    
    # Evaluate individual models
    print("\n--- Model Benchmark Comparison on February Validation Set ---")
    cat_predictor = StatePredictor(ASSETS)
    pred_v_cat = cat_predictor.residual(X_val, "speed")
    
    lgb_speed = lgb.Booster(model_file=str(OUT_MODELS / "lgb_speed_residual_deep127.txt"))
    pred_v_lgb = lgb_speed.predict(X_val, num_threads=6)
    
    pred_v_xgb = model_xgb.predict(X_val)
    
    rmse_cat = float(np.sqrt(np.mean((yv_val - pred_v_cat) ** 2)))
    rmse_lgb = float(np.sqrt(np.mean((yv_val - pred_v_lgb) ** 2)))
    rmse_xgb = float(np.sqrt(np.mean((yv_val - pred_v_xgb) ** 2)))
    
    print(f"1. CatBoost Benchmark Speed RMSE:  {rmse_cat:.4f} km/h")
    print(f"2. Deep LightGBM Speed RMSE:       {rmse_lgb:.4f} km/h")
    print(f"3. XGBoost Depth-8 Speed RMSE:     {rmse_xgb:.4f} km/h")
    
    # 2-way blend (S18 reference: 70% CB + 30% LGB)
    blend_2way = 0.70 * pred_v_cat + 0.30 * pred_v_lgb
    rmse_2way = float(np.sqrt(np.mean((yv_val - blend_2way) ** 2)))
    print(f"\nS18 Proven 2-way Blend (70% CB + 30% LGB): {rmse_2way:.4f} km/h (Delta = {rmse_2way - rmse_cat:+.4f})")
    
    # Numerical Optimization of 3-Way Weights on Simplex
    print("\n--- Optimizing 3-Way Ensemble Weights (CatBoost + LightGBM + XGBoost) ---")
    
    def loss_func(weights):
        w_c, w_l, w_x = weights
        pred = w_c * pred_v_cat + w_l * pred_v_lgb + w_x * pred_v_xgb
        return np.sqrt(np.mean((yv_val - pred) ** 2))
    
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    init_weights = [0.65, 0.25, 0.10]
    
    opt = minimize(loss_func, init_weights, method="SLSQP", bounds=bounds, constraints=cons)
    w_cb_opt, w_lgb_opt, w_xgb_opt = opt.x
    rmse_3way = opt.fun
    
    print(f"Optimal 3-Way Weights:")
    print(f"  * CatBoost: {w_cb_opt*100:.1f}%")
    print(f"  * LightGBM: {w_lgb_opt*100:.1f}%")
    print(f"  * XGBoost:  {w_xgb_opt*100:.1f}%")
    print(f"3-Way Optimal Speed RMSE: {rmse_3way:.4f} km/h")
    print(f"Delta vs CatBoost Benchmark: {rmse_3way - rmse_cat:+.4f} km/h")
    print(f"Delta vs S18 2-Way Blend:   {rmse_3way - rmse_2way:+.4f} km/h")
    
    # Save ensemble weights report
    report = {
        "rmse_catboost": rmse_cat,
        "rmse_lightgbm": rmse_lgb,
        "rmse_xgboost": rmse_xgb,
        "rmse_2way_s18": rmse_2way,
        "rmse_3way_optimal": rmse_3way,
        "weights": {
            "catboost": round(float(w_cb_opt), 4),
            "lightgbm": round(float(w_lgb_opt), 4),
            "xgboost": round(float(w_xgb_opt), 4),
        },
    }
    (OUT_MODELS / "speed_tri_ensemble_report.json").write_text(json.dumps(report, indent=2))
    print(f"Saved report -> outputs/models/speed_tri_ensemble_report.json")


if __name__ == "__main__":
    main()
