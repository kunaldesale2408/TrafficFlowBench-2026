"""Train deep LightGBM flow residual model (num_leaves=127).
Matches the capacity and feature representations of lgb_speed_residual_deep127.
Evaluates flow RMSE against CatBoost benchmark.
Outputs:
  outputs/models/lgb_flow_residual_deep127.txt
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

ROOT = Path("trafficflowbench").resolve()
DATA = ROOT / "data" / "kaggle_public"
ASSETS = ROOT / "kaggle_research" / "anubis_assets"
OUT_MODELS = ROOT / "outputs" / "models"
OUT_MODELS.mkdir(parents=True, exist_ok=True)

os.environ["TFB_DATA_ROOT"] = str(DATA)
os.environ["TFB_OUTPUT_ROOT"] = str(ROOT / "scratch")
os.environ["TFB_THREADS"] = "6"
os.environ["TFB_PREDICT_DEVICE"] = "CPU"

sys.path.insert(0, str(ASSETS))
import state_model as base
import prepare_state
from state_inference import StatePredictor


def collect_dataset(samples_per_day: int = 1200):
    t0 = time.time()
    print(f"Collecting flow training and validation datasets ({samples_per_day} samples/day)...")
    
    train_x, train_yq = [], []
    val_x, val_yq = [], []
    val_bq, val_lanes = [], []
    
    panels = base.all_panels()
    train_months = base.TRAIN_MONTHS
    val_month = base.VALID_MONTH
    
    for pi, name in enumerate(panels):
        p = base.Panel(name, pi)
        # Training months
        for month in train_months:
            times, v, q, e = base.matrices(base.month_frame(name, month), p)
            mask, regime = base.simulated_mask(name, times, v, q, e)
            rows, cols = base.sampled_targets(mask, times, samples_per_day)
            if len(rows) == 0:
                continue
            X, _, bq = prepare_state.features(
                p, pi, times, np.where(mask, np.nan, v), np.where(mask, np.nan, q), e, rows, cols, regime
            )
            rq = q[rows, cols] / p.lanes[cols] - bq
            valid_idx = np.isfinite(rq)
            train_x.append(X[valid_idx])
            train_yq.append(rq[valid_idx])
        
        # Validation month (February 2031-02)
        times, v, q, e = base.matrices(base.month_frame(name, val_month), p)
        mask, regime = base.simulated_mask(name, times, v, q, e)
        rows, cols = base.sampled_targets(mask, times, 1500)
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
            
        print(f"  Panel {name}: loaded train and validation")
        
    X_train = np.vstack(train_x)
    yq_train = np.concatenate(train_yq)
    
    X_val = np.vstack(val_x)
    yq_val = np.concatenate(val_yq)
    bq_val = np.concatenate(val_bq)
    lanes_val = np.concatenate(val_lanes)
    
    print(f"Data collection completed in {time.time()-t0:.1f}s.")
    print(f"Train set: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"Val set:   {X_val.shape[0]:,} samples")
    return (X_train, yq_train), (X_val, yq_val, bq_val, lanes_val)


def main():
    print("=" * 70)
    print(">>> TRAINING DEEP LIGHTGBM FLOW RESIDUAL MODEL (num_leaves=127) <<<")
    print("=" * 70)
    
    (X_train, yq_train), (X_val, yq_val, bq_val, lanes_val) = collect_dataset(samples_per_day=1200)
    
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "num_leaves": 127,
        "min_child_samples": 25,
        "subsample": 0.75,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.5,
        "reg_lambda": 3.0,
        "n_jobs": 6,
        "verbose": -1,
        "random_state": 20261026,
    }
    
    print("\n--- Training Deep LightGBM Flow Model ---")
    t0 = time.time()
    ds_train_q = lgb.Dataset(X_train, label=yq_train, free_raw_data=False)
    ds_val_q = lgb.Dataset(X_val, label=yq_val, reference=ds_train_q, free_raw_data=False)
    
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=True), lgb.log_evaluation(period=100)]
    model_q = lgb.train(
        params,
        ds_train_q,
        valid_sets=[ds_train_q, ds_val_q],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    print(f"Deep Flow model trained in {time.time()-t0:.1f}s, best iter: {model_q.best_iteration}")
    save_path = OUT_MODELS / "lgb_flow_residual_deep127.txt"
    model_q.save_model(str(save_path))
    print(f"Saved model -> {save_path.name}")
    
    # Evaluate CatBoost vs New Deep LightGBM vs Blend
    print("\nLoading CatBoost benchmark predictions...")
    cat_predictor = StatePredictor(ASSETS)
    pred_q_cat = cat_predictor.residual(X_val, "flow")
    pred_q_lgb = model_q.predict(X_val, num_threads=6)
    
    rmse_cat = np.sqrt(np.mean((yq_val - pred_q_cat) ** 2))
    rmse_lgb = np.sqrt(np.mean((yq_val - pred_q_lgb) ** 2))
    print(f"CatBoost Benchmark Flow RMSE: {rmse_cat:.4f} vph/lane")
    print(f"Deep LightGBM Flow RMSE:      {rmse_lgb:.4f} vph/lane")
    
    print("\n--- Grid Search Blend Weights ---")
    best_rmse = rmse_cat
    best_w = 0.0
    for w_lgb in [0.05, 0.10, 0.12, 0.15, 0.20, 0.25]:
        w_cb = 1.0 - w_lgb
        blend = w_cb * pred_q_cat + w_lgb * pred_q_lgb
        r = np.sqrt(np.mean((yq_val - blend) ** 2))
        marker = " *** BEST ***" if r < best_rmse else ""
        if r < best_rmse:
            best_rmse = r
            best_w = w_lgb
        print(f"{w_cb*100:.0f}% CB + {w_lgb*100:.0f}% Deep-LGB: Flow RMSE = {r:.4f} vph/lane (Delta = {r - rmse_cat:+.4f}){marker}")
        
    print("-" * 65)
    print(f"OPTIMAL FLOW BLEND: {100-best_w*100:.0f}% CB + {best_w*100:.0f}% Deep-LGB -> RMSE: {best_rmse:.4f} vph/lane")


if __name__ == "__main__":
    main()
