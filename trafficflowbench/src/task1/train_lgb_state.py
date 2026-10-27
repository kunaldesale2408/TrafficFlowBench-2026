"""Train spatiotemporal LightGBM residual models on the exact 94-feature representation.

Evaluates:
  - CatBoost validation RMSE
  - LightGBM validation RMSE
  - Blended ensemble validation RMSE across multiple weights
Saves models for ensembling into Task 1 state predictions.
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

# Setup paths and environment
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


def collect_dataset(samples_per_day: int = 750):
    t0 = time.time()
    print(f"Collecting training and validation datasets ({samples_per_day} samples/day)...")
    
    train_x, train_yv, train_yq = [], [], []
    val_x, val_yv, val_yq = [], [], []
    val_bv, val_bq, val_lanes = [], [], []
    
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
            X, bv, bq = prepare_state.features(
                p, pi, times, np.where(mask, np.nan, v), np.where(mask, np.nan, q), e, rows, cols, regime
            )
            rv = v[rows, cols] - bv
            rq = q[rows, cols] / p.lanes[cols] - bq
            
            # Filter finite targets
            valid_idx = np.isfinite(rv) & np.isfinite(rq)
            train_x.append(X[valid_idx])
            train_yv.append(rv[valid_idx])
            train_yq.append(rq[valid_idx])
        
        # Validation month (February 2031-02)
        times, v, q, e = base.matrices(base.month_frame(name, val_month), p)
        mask, regime = base.simulated_mask(name, times, v, q, e)
        rows, cols = base.sampled_targets(mask, times, 1500)
        if len(rows) > 0:
            X, bv, bq = prepare_state.features(
                p, pi, times, np.where(mask, np.nan, v), np.where(mask, np.nan, q), e, rows, cols, regime
            )
            rv = v[rows, cols] - bv
            rq = q[rows, cols] / p.lanes[cols] - bq
            valid_idx = np.isfinite(rv) & np.isfinite(rq)
            val_x.append(X[valid_idx])
            val_yv.append(rv[valid_idx])
            val_yq.append(rq[valid_idx])
            val_bv.append(bv[valid_idx])
            val_bq.append(bq[valid_idx])
            val_lanes.append(p.lanes[cols][valid_idx])
        
        print(f"  Processed {name} ({pi+1}/{len(panels)}) in {time.time()-t0:.1f}s")
    
    print("Concatenating matrices...")
    X_train = np.concatenate(train_x)
    yv_train = np.concatenate(train_yv)
    yq_train = np.concatenate(train_yq)
    
    X_val = np.concatenate(val_x)
    yv_val = np.concatenate(val_yv)
    yq_val = np.concatenate(val_yq)
    bv_val = np.concatenate(val_bv)
    bq_val = np.concatenate(val_bq)
    lanes_val = np.concatenate(val_lanes)
    
    print(f"Data ready: Train = {len(X_train):,} rows, Val = {len(X_val):,} rows, Features = {X_train.shape[1]}")
    print(f"Total time for feature collection: {time.time()-t0:.1f}s")
    return (X_train, yv_train, yq_train), (X_val, yv_val, yq_val, bv_val, bq_val, lanes_val)


def train_lgb_models(train_data, val_data):
    X_train, yv_train, yq_train = train_data
    X_val, yv_val, yq_val, bv_val, bq_val, lanes_val = val_data
    
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 1200,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 30,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "n_jobs": 6,
        "verbose": -1,
        "random_state": 20261026,
    }
    
    print("\n--- Training LightGBM Speed Residual Model ---")
    t0 = time.time()
    ds_train_v = lgb.Dataset(X_train, label=yv_train, free_raw_data=False)
    ds_val_v = lgb.Dataset(X_val, label=yv_val, reference=ds_train_v, free_raw_data=False)
    
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=True), lgb.log_evaluation(period=100)]
    model_v = lgb.train(
        params,
        ds_train_v,
        valid_sets=[ds_train_v, ds_val_v],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    print(f"Speed model trained in {time.time()-t0:.1f}s, best iter: {model_v.best_iteration}")
    model_v.save_model(str(OUT_MODELS / "lgb_speed_residual_94f.txt"))
    del ds_train_v, ds_val_v
    gc.collect()
    
    print("\n--- Training LightGBM Flow Residual Model ---")
    t0 = time.time()
    ds_train_q = lgb.Dataset(X_train, label=yq_train, free_raw_data=False)
    ds_val_q = lgb.Dataset(X_val, label=yq_val, reference=ds_train_q, free_raw_data=False)
    
    model_q = lgb.train(
        params,
        ds_train_q,
        valid_sets=[ds_train_q, ds_val_q],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    print(f"Flow model trained in {time.time()-t0:.1f}s, best iter: {model_q.best_iteration}")
    model_q.save_model(str(OUT_MODELS / "lgb_flow_residual_94f.txt"))
    del ds_train_q, ds_val_q
    gc.collect()
    
    return model_v, model_q


def evaluate_ensemble(model_v, model_q, val_data):
    X_val, yv_val, yq_val, bv_val, bq_val, lanes_val = val_data
    
    print("\n--- Running CatBoost Validation Inference ---")
    t0 = time.time()
    sp = StatePredictor(ASSETS)
    pred_v_cat = sp.residual(X_val, "speed")
    pred_q_cat = sp.residual(X_val, "flow")
    print(f"CatBoost inference completed in {time.time()-t0:.1f}s")
    
    print("\n--- Running LightGBM Validation Inference ---")
    pred_v_lgb = model_v.predict(X_val, num_threads=6)
    pred_q_lgb = model_q.predict(X_val, num_threads=6)
    
    # Competition metric calculation:
    # S_state = 1.0 - 0.5 * (RMSE_speed / 25.0) - 0.5 * (RMSE_flow_lane / 600.0)
    # Target residuals are yv_val and yq_val
    def calc_metrics(p_v, p_q):
        rmse_v = np.sqrt(np.mean((yv_val - p_v) ** 2))
        rmse_q = np.sqrt(np.mean((yq_val - p_q) ** 2))
        score = 1.0 - 0.5 * (rmse_v / 25.0) - 0.5 * (rmse_q / 600.0)
        return rmse_v, rmse_q, score
    
    print("\n" + "=" * 70)
    print("VALIDATION SCORE COMPARISON (February 2031-02)")
    print("=" * 70)
    print(f"{'Model Configuration':<30} | {'Speed RMSE':<10} | {'Flow/Lane RMSE':<14} | {'S_state Score':<12}")
    print("-" * 70)
    
    rmse_v_cat, rmse_q_cat, score_cat = calc_metrics(pred_v_cat, pred_q_cat)
    print(f"{'CatBoost (Benchmark)':<30} | {rmse_v_cat:<10.4f} | {rmse_q_cat:<14.4f} | {score_cat:<12.5f}")
    
    rmse_v_lgb, rmse_q_lgb, score_lgb = calc_metrics(pred_v_lgb, pred_q_lgb)
    print(f"{'LightGBM (Ours)':<30} | {rmse_v_lgb:<10.4f} | {rmse_q_lgb:<14.4f} | {score_lgb:<12.5f}")
    
    best_weight = 0.0
    best_score = score_cat
    
    for w_lgb in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        w_cat = 1.0 - w_lgb
        blend_v = w_cat * pred_v_cat + w_lgb * pred_v_lgb
        blend_q = w_cat * pred_q_cat + w_lgb * pred_q_lgb
        r_v, r_q, sc = calc_metrics(blend_v, blend_q)
        delta = sc - score_cat
        marker = " *** BEST ***" if sc > best_score else ""
        if sc > best_score:
            best_score = sc
            best_weight = w_lgb
        print(f"{f'Blend ({int(w_cat*100)}% CB + {int(w_lgb*100)}% LGB)':<30} | {r_v:<10.4f} | {r_q:<14.4f} | {sc:<12.5f} ({delta:+.5f}){marker}")
    
    print("-" * 70)
    print(f"Optimal Blend Weight: {100-int(best_weight*100)}% CatBoost + {int(best_weight*100)}% LightGBM (Score: {best_score:.5f})")
    
    # Save validation report
    report = {
        "catboost_rmse_speed": float(rmse_v_cat),
        "catboost_rmse_flow": float(rmse_q_cat),
        "catboost_s_state": float(score_cat),
        "lgb_rmse_speed": float(rmse_v_lgb),
        "lgb_rmse_flow": float(rmse_q_lgb),
        "lgb_s_state": float(score_lgb),
        "optimal_lgb_weight": float(best_weight),
        "optimal_s_state": float(best_score),
        "gain": float(best_score - score_cat),
    }
    with open(OUT_MODELS / "validation_ensemble_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return best_weight


def main():
    print("=" * 70)
    print(">>> TASK 1 STATE RECONSTRUCTION: LIGHTGBM TRAINING & ENSEMBLING <<<")
    print("=" * 70)
    train_data, val_data = collect_dataset(samples_per_day=750)
    model_v, model_q = train_lgb_models(train_data, val_data)
    best_weight = evaluate_ensemble(model_v, model_q, val_data)
    print(f"\nTraining and validation complete. Optimal LGB weight: {best_weight:.2f}")


if __name__ == "__main__":
    main()
