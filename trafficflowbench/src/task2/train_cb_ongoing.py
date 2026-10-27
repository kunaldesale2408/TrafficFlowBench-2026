"""Train complementary CatBoost Ongoing Queue Classifier to ensemble with LightGBM.

Evaluates:
  - Original LightGBM Ongoing IoU on validation windows
  - CatBoost Ongoing IoU on validation windows
  - 50/50 Blended Probability Ongoing IoU
Saves:
  outputs/models/ongoing_cb.cbm
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path("trafficflowbench").resolve()
DATA = ROOT / "data" / "kaggle_public"
ASSETS = ROOT / "kaggle_research" / "anubis_assets"
OUT_MODELS = ROOT / "outputs" / "models"
OUT_MODELS.mkdir(parents=True, exist_ok=True)

os.environ["TFB_DATA_ROOT"] = str(DATA)
os.environ["TFB_OUTPUT_ROOT"] = str(ROOT / "scratch")
os.environ["TFB_THREADS"] = "6"

sys.path.insert(0, str(ASSETS))
import queue_ongoing as q_module


def main():
    print("=" * 70)
    print(">>> TRAINING CATBOOST ONGOING QUEUE CLASSIFIER FOR ENSEMBLING <<<")
    print("=" * 70)

    # 1. Collect dataset
    t0 = time.time()
    X, y, weights, valid_windows, feature_count = q_module.collect()
    print(f"Data collected in {time.time()-t0:.1f}s: X={X.shape}, y={y.shape}, features={feature_count}")

    # Separate ongoing validation windows
    ongoing_val = [w for w in valid_windows if w.condition == "queue_ongoing"]
    print(f"Ongoing validation windows: {len(ongoing_val)}")

    X_val = np.vstack([w.X[w.eligible] for w in ongoing_val])
    y_val = np.concatenate([w.y[w.eligible] for w in ongoing_val]).astype(np.uint8)
    w_val = np.concatenate([np.full(w.eligible.sum(), 1000 / w.eligible.sum(), dtype=np.float32) for w in ongoing_val])

    # 2. Load existing LightGBM model and score validation
    orig_lgb = lgb.Booster(model_file=str(ASSETS / "ongoing_lgb.txt"))
    probs_lgb = [orig_lgb.predict(w.X, num_threads=6) for w in ongoing_val]

    def eval_iou(probs_list, threshold):
        ious = []
        for w, p in zip(ongoing_val, probs_list):
            pred = (p[w.eligible] >= threshold).astype(bool)
            true = w.y[w.eligible].astype(bool)
            intersection = (pred & true).sum()
            union = (pred | true).sum()
            if union > 0:
                ious.append(intersection / union)
        return float(np.mean(ious))

    best_lgb_iou = max([eval_iou(probs_lgb, t) for t in np.arange(0.30, 0.60, 0.025)])
    print(f"Original LightGBM Best Validation Ongoing IoU: {best_lgb_iou:.5f}")

    # 3. Train CatBoost model
    print("\nTraining CatBoostClassifier on ongoing queue features...")
    t_cb = time.time()
    cb_model = CatBoostClassifier(
        iterations=800,
        learning_rate=0.045,
        depth=6,
        l2_leaf_reg=5.0,
        loss_function="Logloss",
        eval_metric="Logloss",
        thread_count=6,
        random_seed=20261026,
        verbose=100,
    )
    cb_model.fit(
        X, y,
        sample_weight=weights,
        eval_set=(X_val, y_val),
        early_stopping_rounds=60,
        verbose=100,
    )
    print(f"CatBoost trained in {time.time()-t_cb:.1f}s, best iteration: {cb_model.get_best_iteration()}")

    # 4. Score CatBoost and Blend
    probs_cb = [cb_model.predict_proba(w.X, thread_count=6)[:, 1] for w in ongoing_val]
    best_cb_iou = max([eval_iou(probs_cb, t) for t in np.arange(0.30, 0.60, 0.025)])
    print(f"CatBoost Best Validation Ongoing IoU: {best_cb_iou:.5f}")

    print("\n" + "=" * 70)
    print("ENSEMBLE PROBABILITY AVERAGE EVALUATION (Validation Windows)")
    print("=" * 70)
    print(f"{'Configuration':<35} | {'Threshold':<10} | {'Ongoing IoU':<12}")
    print("-" * 70)

    # 50/50 blend
    probs_blend = [0.5 * p_l + 0.5 * p_c for p_l, p_c in zip(probs_lgb, probs_cb)]

    best_blend_iou = 0.0
    best_thresh = 0.45
    for t in [0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.50]:
        iou = eval_iou(probs_blend, t)
        delta = iou - best_lgb_iou
        marker = " *** BEST ***" if iou > best_blend_iou else ""
        if iou > best_blend_iou:
            best_blend_iou = iou
            best_thresh = t
        print(f"{'50% LGBM + 50% CatBoost':<35} | {t:<10.3f} | {iou:<12.5f} ({delta:+.5f}){marker}")

    print("-" * 70)
    print(f"Optimal Ongoing Ensemble: 50% LGB + 50% CB @ threshold {best_thresh:.3f} (IoU: {best_blend_iou:.5f})")

    # Save CatBoost model
    out_cb_path = OUT_MODELS / "ongoing_cb.cbm"
    cb_model.save_model(str(out_cb_path))
    print(f"Model saved to {out_cb_path.resolve()}")

    report = {
        "lgb_best_iou": best_lgb_iou,
        "cb_best_iou": best_cb_iou,
        "blend_best_iou": best_blend_iou,
        "best_threshold": best_thresh,
        "iou_gain": best_blend_iou - best_lgb_iou,
    }
    with open(OUT_MODELS / "ongoing_ensemble_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
