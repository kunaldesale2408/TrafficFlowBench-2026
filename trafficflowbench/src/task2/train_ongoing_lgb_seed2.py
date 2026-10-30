"""Train Diversified Companion LightGBM Ongoing Queue Model (Seed 1024).
Creates an independent LightGBM ongoing model to enable multi-seed probability ensembling:
  - Architecture: max_depth=7, num_leaves=24, min_data_in_leaf=300
  - Regularization: lambda_l2=10.0, feature_fraction=0.80, bagging_fraction=0.80
  - Seed: 1024
Outputs:
  outputs/models/ongoing_lgb_seed1024.txt
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

ROOT = Path("trafficflowbench").resolve()
DATA = ROOT / "data" / "kaggle_public"
ASSETS = ROOT / "kaggle_research" / "anubis_assets"
MODELS_DIR = ROOT / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

os.environ["TFB_DATA_ROOT"] = str(DATA)
os.environ["TFB_OUTPUT_ROOT"] = str(MODELS_DIR)
os.environ["TFB_THREADS"] = "6"

sys.path.insert(0, str(ASSETS))
import queue_ongoing as qo

def main():
    t0 = time.time()
    print("=" * 75)
    print(">>> TRAINING COMPANION LIGHTGBM ONGOING QUEUE MODEL (SEED 1024) <<<")
    print("=" * 75)

    print("Collecting training and validation data across all corridors...")
    X, y, weights, windows, feature_count = qo.collect()
    print(f"Data collected in {time.time()-t0:.1f}s: X={X.shape}, pos={y.mean():.4f}")

    X_val = np.vstack([w.X[w.eligible] for w in windows])
    y_val = np.concatenate([w.y[w.eligible] for w in windows]).astype(np.uint8)
    w_val = np.concatenate([np.full(w.eligible.sum(), 1000 / w.eligible.sum(), dtype=np.float32) for w in windows])

    train_set = lgb.Dataset(X, label=y, weight=weights, free_raw_data=True)
    valid_set = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=train_set, free_raw_data=True)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.035,
        "num_leaves": 24,
        "max_depth": 7,
        "min_data_in_leaf": 300,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "lambda_l2": 10.0,
        "max_bin": 127,
        "num_threads": 6,
        "verbosity": -1,
        "seed": 1024,
    }

    print("\nTraining LightGBM model with seed 1024...")
    t_train = time.time()
    model = lgb.train(
        params,
        train_set,
        num_boost_round=800,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(60, verbose=True), lgb.log_evaluation(50)],
    )
    print(f"Training finished in {time.time()-t_train:.1f}s. Best iteration: {model.best_iteration}")

    out_model_path = MODELS_DIR / "ongoing_lgb_seed1024.txt"
    model.save_model(str(out_model_path))
    print(f"Saved model to {out_model_path}")

    # Evaluate standalone vs ensembled with base model on validation windows
    print("\nEvaluating on February 2031 validation windows...")
    base_model = lgb.Booster(model_file=str(ASSETS / "ongoing_lgb.txt"))
    probs_base = [base_model.predict(w.X, num_threads=6) for w in windows]
    probs_seed2 = [model.predict(w.X, num_threads=6) for w in windows]
    probs_ensemble = [0.5 * p1 + 0.5 * p2 for p1, p2 in zip(probs_base, probs_seed2)]

    trials = []
    for th in [0.40, 0.42, 0.44, 0.45, 0.46, 0.47, 0.48, 0.485, 0.49, 0.50]:
        ov_base, _, ong_base = qo.score_windows(windows, probs_base, th, 0)
        ov_s2, _, ong_s2 = qo.score_windows(windows, probs_seed2, th, 0)
        ov_ens, _, ong_ens = qo.score_windows(windows, probs_ensemble, th, 0)
        trials.append({
            "threshold": th,
            "ong_base": ong_base,
            "ong_s2": ong_s2,
            "ong_ensemble": ong_ens,
            "gain_vs_base": ong_ens - ong_base,
        })

    report_df = pd.DataFrame(trials)
    print("\n" + "=" * 75)
    print(">>> VALIDATION ONGOING IOU: BASE vs SEED1024 vs 50/50 ENSEMBLE <<<")
    print("=" * 75)
    print(report_df.round(5).to_string(index=False))

    best_row = report_df.loc[report_df["ong_ensemble"].idxmax()]
    print(f"\nBest Ensemble Threshold: {best_row['threshold']} | Ongoing IoU: {best_row['ong_ensemble']:.5f} (Gain: {best_row['gain_vs_base']:+.5f})")

    report_json = MODELS_DIR / "ongoing_lgb_seed1024_report.json"
    report_json.write_text(json.dumps(trials, indent=2), encoding="utf-8")
    print(f"Report saved to {report_json}")
    print(f"Total time elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
