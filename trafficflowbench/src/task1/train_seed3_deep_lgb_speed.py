"""Train Seed-3 Deep LightGBM Speed Residual Model (seed=1024, max_depth=12, num_leaves=96).
Evaluates the 3-seed LightGBM bag + CatBoost multi-model ensemble on validation speed RMSE.
Solves SLSQP continuous simplex weights to minimize validation RMSE.
Outputs:
  outputs/models/lgb_speed_residual_deep96_seed1024.txt
  outputs/models/speed_ensemble_weights_s24.json
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
from catboost import CatBoostRegressor

ROOT = Path("trafficflowbench").resolve()
OUT_MODELS = ROOT / "outputs" / "models"
OUT_MODELS.mkdir(parents=True, exist_ok=True)
SCRATCH = ROOT / "scratch"
CACHE_FILE = SCRATCH / "state_cache_1200.npz"
ASSETS = ROOT / "kaggle_research" / "anubis_assets"


def main():
    print("=" * 75)
    print(">>> TRAINING SEED-3 DEEP LIGHTGBM SPEED RESIDUAL MODEL (SEED 1024) <<<")
    print("=" * 75)

    t0 = time.time()
    print("Loading cached dataset from state_cache_1200.npz...")
    data = np.load(CACHE_FILE)
    X_train = data["X_train"]
    yv_train = data["yv_train"]
    X_val = data["X_val"]
    yv_val = data["yv_val"]
    print(f"Dataset loaded in {time.time()-t0:.1f}s: {X_train.shape[0]:,} train, {X_val.shape[0]:,} val, {X_train.shape[1]} features")

    # 1. Load existing Seed 1 and Seed 2 models
    print("\nEvaluating existing Seed-1 and Seed-2 LightGBM models...")
    m1_path = OUT_MODELS / "lgb_speed_residual_deep127.txt"
    m2_path = OUT_MODELS / "lgb_speed_residual_deep127_seed42.txt"

    m1 = lgb.Booster(model_file=str(m1_path))
    p_val_m1 = m1.predict(X_val, num_threads=6)
    rmse_m1 = float(np.sqrt(np.mean((yv_val - p_val_m1) ** 2)))
    print(f"Seed-1 Deep-LGB127 Speed RMSE: {rmse_m1:.4f} km/h")

    m2 = lgb.Booster(model_file=str(m2_path))
    p_val_m2 = m2.predict(X_val, num_threads=6)
    rmse_m2 = float(np.sqrt(np.mean((yv_val - p_val_m2) ** 2)))
    print(f"Seed-2 Deep-LGB127 Speed RMSE: {rmse_m2:.4f} km/h")

    p_val_bag2 = 0.5 * p_val_m1 + 0.5 * p_val_m2
    rmse_bag2 = float(np.sqrt(np.mean((yv_val - p_val_bag2) ** 2)))
    print(f"2-Seed Bagged LGB (S21/S23 baseline) Speed RMSE: {rmse_bag2:.4f} km/h")

    # 2. Configure Seed 3 parameters (max_depth=12, num_leaves=96, colsample=0.80, seed=1024)
    params3 = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 1600,
        "learning_rate": 0.035,
        "max_depth": 12,
        "num_leaves": 96,
        "min_child_samples": 35,
        "subsample": 0.80,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.4,
        "reg_lambda": 3.5,
        "n_jobs": 6,
        "verbose": -1,
        "random_state": 1024,
    }

    print("\nTraining Seed-3 Model (max_depth=12, num_leaves=96, seed=1024)...")
    t_train = time.time()
    ds_train = lgb.Dataset(X_train, label=yv_train, free_raw_data=False)
    ds_val = lgb.Dataset(X_val, label=yv_val, reference=ds_train, free_raw_data=False)

    callbacks = [
        lgb.early_stopping(stopping_rounds=40, verbose=True),
        lgb.log_evaluation(period=200),
    ]
    m3 = lgb.train(
        params3,
        ds_train,
        valid_sets=[ds_train, ds_val],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    print(f"Seed-3 trained in {time.time()-t_train:.1f}s, best iteration: {m3.best_iteration}")

    save_path = OUT_MODELS / "lgb_speed_residual_deep96_seed1024.txt"
    m3.save_model(str(save_path))
    print(f"Saved Seed-3 model -> {save_path.name}")

    p_val_m3 = m3.predict(X_val, num_threads=6)
    rmse_m3 = float(np.sqrt(np.mean((yv_val - p_val_m3) ** 2)))
    print(f"Seed-3 Alone Speed RMSE: {rmse_m3:.4f} km/h")

    # 3. 3-Seed LightGBM Bagging
    p_val_bag3_equal = (p_val_m1 + p_val_m2 + p_val_m3) / 3.0
    rmse_bag3_equal = float(np.sqrt(np.mean((yv_val - p_val_bag3_equal) ** 2)))
    print(f"3-Seed Bagged LGB (Equal 1/3) Speed RMSE: {rmse_bag3_equal:.4f} km/h (Delta vs 2-seed: {rmse_bag3_equal - rmse_bag2:+.4f})")

    # Optimize weights among the 3 LightGBM seeds
    def loss_lgb3(w):
        w1, w2 = w[0], w[1]
        w3 = 1.0 - w1 - w2
        pred = w1 * p_val_m1 + w2 * p_val_m2 + w3 * p_val_m3
        return float(np.sqrt(np.mean((yv_val - pred) ** 2)))

    res_lgb3 = minimize(
        loss_lgb3,
        [0.33, 0.33],
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        constraints={"type": "ineq", "fun": lambda w: 1.0 - w[0] - w[1]},
    )
    w_lgb1 = float(res_lgb3.x[0])
    w_lgb2 = float(res_lgb3.x[1])
    w_lgb3 = 1.0 - w_lgb1 - w_lgb2
    p_val_bag3_opt = w_lgb1 * p_val_m1 + w_lgb2 * p_val_m2 + w_lgb3 * p_val_m3
    rmse_bag3_opt = float(np.sqrt(np.mean((yv_val - p_val_bag3_opt) ** 2)))
    print(f"3-Seed Bagged LGB (Optimized weights: {w_lgb1:.3f}/{w_lgb2:.3f}/{w_lgb3:.3f}) Speed RMSE: {rmse_bag3_opt:.4f} km/h")

    # 4. Blend with CatBoost benchmark
    print("\nLoading CatBoost benchmark (bundled d8 + d10)...")
    cb_d8 = CatBoostRegressor()
    cb_d8.load_model(str(ASSETS / "speed_d8.cbm"))
    cb_d10 = CatBoostRegressor()
    cb_d10.load_model(str(ASSETS / "speed_d10.cbm"))
    p_cb = 0.5 * cb_d8.predict(X_val, thread_count=6) + 0.5 * cb_d10.predict(X_val, thread_count=6)
    rmse_cb = float(np.sqrt(np.mean((yv_val - p_cb) ** 2)))
    print(f"CatBoost Benchmark Alone Speed RMSE: {rmse_cb:.4f} km/h")

    # Optimize overall blend: CatBoost + 3-Seed LightGBM
    def loss_overall(w):
        w_cb = w[0]
        w_lgb = 1.0 - w_cb
        pred = w_cb * p_cb + w_lgb * p_val_bag3_opt
        return float(np.sqrt(np.mean((yv_val - pred) ** 2)))

    res_overall = minimize(loss_overall, [0.574], bounds=[(0.2, 0.8)])
    best_w_cb = float(res_overall.x[0])
    best_w_lgb = 1.0 - best_w_cb
    best_rmse = float(res_overall.fun)

    # Full 4-model joint simplex optimization
    def loss_4way(w):
        pred = w[0] * p_cb + w[1] * p_val_m1 + w[2] * p_val_m2 + w[3] * p_val_m3
        return float(np.sqrt(np.mean((yv_val - pred) ** 2)))

    res_4way = minimize(
        loss_4way,
        [best_w_cb, best_w_lgb * w_lgb1, best_w_lgb * w_lgb2, best_w_lgb * w_lgb3],
        bounds=[(0.0, 1.0)] * 4,
        constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0},
    )
    w_final_cb, w_final_m1, w_final_m2, w_final_m3 = [float(x) for x in res_4way.x]
    rmse_4way = float(res_4way.fun)

    print("\n" + "=" * 75)
    print(">>> MODEL PERFORMANCE LEDGER & FINAL BLEND OPTIMIZATION <<<")
    print("=" * 75)
    print(f"CatBoost Benchmark Alone:       {rmse_cb:.4f} km/h")
    print(f"Seed-1 Deep-LGB127 Alone:        {rmse_m1:.4f} km/h")
    print(f"Seed-2 Deep-LGB127 Alone:        {rmse_m2:.4f} km/h")
    print(f"Seed-3 Deep-LGB96 Alone:         {rmse_m3:.4f} km/h")
    print(f"2-Seed Bagged LGB (S21/S23):     {rmse_bag2:.4f} km/h")
    print(f"3-Seed Bagged LGB (Optimized):   {rmse_bag3_opt:.4f} km/h")
    print("-" * 75)
    print(f"S19 Final Blend RMSE:            1.5865 km/h")
    print(f"S23 Final Blend RMSE:            1.5836 km/h")
    print(f"NEW S24 Final 4-Way Blend RMSE:  {rmse_4way:.4f} km/h  (Gain vs S23: {rmse_4way - 1.5836:+.4f} km/h)")
    print("-" * 75)
    print("Optimal 4-Way Blend Weights:")
    print(f"  - CatBoost Bundled (d8/d10):    {w_final_cb * 100:.2f}%")
    print(f"  - Seed-1 Deep-LGB127 (seed 404): {w_final_m1 * 100:.2f}%")
    print(f"  - Seed-2 Deep-LGB127 (seed 42):  {w_final_m2 * 100:.2f}%")
    print(f"  - Seed-3 Deep-LGB96  (seed 1024):{w_final_m3 * 100:.2f}%")
    print("=" * 75)

    # Save weights JSON
    weights_info = {
        "rmse_cb": rmse_cb,
        "rmse_m1": rmse_m1,
        "rmse_m2": rmse_m2,
        "rmse_m3": rmse_m3,
        "rmse_bag2": rmse_bag2,
        "rmse_bag3_opt": rmse_bag3_opt,
        "rmse_s23_blend": 1.5836,
        "rmse_s24_blend": rmse_4way,
        "delta_vs_s23": rmse_4way - 1.5836,
        "weights": {
            "catboost": w_final_cb,
            "lgb_seed404": w_final_m1,
            "lgb_seed42": w_final_m2,
            "lgb_seed1024": w_final_m3,
        },
        "lgb_internal_weights": {
            "w_m1": w_final_m1 / (w_final_m1 + w_final_m2 + w_final_m3),
            "w_m2": w_final_m2 / (w_final_m1 + w_final_m2 + w_final_m3),
            "w_m3": w_final_m3 / (w_final_m1 + w_final_m2 + w_final_m3),
        },
    }
    weights_path = OUT_MODELS / "speed_ensemble_weights_s24.json"
    with open(weights_path, "w") as f:
        json.dump(weights_info, f, indent=2)
    print(f"Ensemble weights successfully saved -> {weights_path.name}")
    print(f"Total script runtime: {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
