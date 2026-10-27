"""Train a second diverse seed of Deep LightGBM (seed=42, colsample=0.70, subsample=0.85) on cached dataset.
Evaluates the 2-seed LightGBM bag vs single seed on validation speed RMSE.
Outputs:
  outputs/models/lgb_speed_residual_deep127_seed42.txt
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize

ROOT = Path("trafficflowbench").resolve()
OUT_MODELS = ROOT / "outputs" / "models"
SCRATCH = ROOT / "scratch"
CACHE_FILE = SCRATCH / "state_cache_1200.npz"

def main():
    print("=" * 70)
    print(">>> TRAINING DIVERSE SEED-2 DEEP LIGHTGBM SPEED RESIDUAL MODEL <<<")
    print("=" * 70)

    t0 = time.time()
    print("Loading cached dataset...")
    data = np.load(CACHE_FILE)
    X_train = data["X_train"]
    yv_train = data["yv_train"]
    X_val = data["X_val"]
    yv_val = data["yv_val"]
    print(f"Dataset loaded in {time.time()-t0:.1f}s: {X_train.shape[0]:,} train, {X_val.shape[0]:,} val")

    # Load existing Seed 1 model
    m1 = lgb.Booster(model_file=str(OUT_MODELS / "lgb_speed_residual_deep127.txt"))
    p_val_m1 = m1.predict(X_val, num_threads=6)
    rmse_m1 = float(np.sqrt(np.mean((yv_val - p_val_m1)**2)))
    print(f"Existing Seed-1 Deep-LGB Speed RMSE: {rmse_m1:.4f} km/h")

    # Configure diverse Seed 2 parameters
    params2 = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 1500,
        "learning_rate": 0.035,
        "num_leaves": 127,
        "min_child_samples": 30,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.70,
        "reg_alpha": 0.8,
        "reg_lambda": 4.0,
        "n_jobs": 6,
        "verbose": -1,
        "random_state": 42,
    }

    print("\nTraining Diverse Seed-2 Model...")
    t_train = time.time()
    ds_train = lgb.Dataset(X_train, label=yv_train, free_raw_data=False)
    ds_val = lgb.Dataset(X_val, label=yv_val, reference=ds_train, free_raw_data=False)
    
    callbacks = [lgb.early_stopping(stopping_rounds=40, verbose=True), lgb.log_evaluation(period=200)]
    m2 = lgb.train(
        params2,
        ds_train,
        valid_sets=[ds_train, ds_val],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    print(f"Seed-2 trained in {time.time()-t_train:.1f}s, best iteration: {m2.best_iteration}")

    save_path = OUT_MODELS / "lgb_speed_residual_deep127_seed42.txt"
    m2.save_model(str(save_path))
    print(f"Saved Seed-2 model -> {save_path.name}")

    p_val_m2 = m2.predict(X_val, num_threads=6)
    rmse_m2 = float(np.sqrt(np.mean((yv_val - p_val_m2)**2)))
    print(f"Seed-2 Alone Speed RMSE: {rmse_m2:.4f} km/h")

    # Bagged 50/50 LightGBM
    p_val_bag = 0.5 * p_val_m1 + 0.5 * p_val_m2
    rmse_bag = float(np.sqrt(np.mean((yv_val - p_val_bag)**2)))
    print(f"Bagged (Seed-1 + Seed-2) Speed RMSE: {rmse_bag:.4f} km/h (Delta = {rmse_bag - rmse_m1:+.4f})")

    # Now evaluate blend with CatBoost benchmark
    from catboost import CatBoostRegressor
    assets = Path("trafficflowbench/kaggle_research/anubis_assets")
    cb_d8 = CatBoostRegressor(); cb_d8.load_model(str(assets / "speed_d8.cbm"))
    cb_d10 = CatBoostRegressor(); cb_d10.load_model(str(assets / "speed_d10.cbm"))
    p_cb = 0.5 * cb_d8.predict(X_val, thread_count=6) + 0.5 * cb_d10.predict(X_val, thread_count=6)
    rmse_cb = float(np.sqrt(np.mean((yv_val - p_cb)**2)))

    # Optimal blend between CatBoost and Bagged LightGBM
    def loss(w):
        pred = (1 - w) * p_cb + w * p_val_bag
        return float(np.sqrt(np.mean((yv_val - pred)**2)))

    res = minimize(loss, [0.35], bounds=[(0.1, 0.6)])
    best_w_lgb = float(res.x[0])
    best_rmse = float(res.fun)

    print("-" * 65)
    print(f"CatBoost Benchmark RMSE:    {rmse_cb:.4f} km/h")
    print(f"S19 Blend (CB + Single LGB): 1.5865 km/h")
    print(f"NEW Blend (CB + Bagged LGB): {best_rmse:.4f} km/h (Optimal LGB weight: {best_w_lgb*100:.1f}%)")
    print(f"Variance Reduction Gain:     {best_rmse - 1.5865:+.4f} km/h")

if __name__ == "__main__":
    main()
