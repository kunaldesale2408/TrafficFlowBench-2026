"""Compare Ongoing Queue Candidates and Decision Thresholds:
1. Benchmark 0.87141 Queue (from submission.csv, 10,596 positives)
2. S24 Queue (from submission_s24.csv, 10,084 positives)
3. Soft Blend Candidates:
   - p_s24 thresholded at 0.45, 0.46, 0.47, 0.48, 0.485
   - 0.5 * p_s24 + 0.5 * p_lgb at 0.45, 0.46, 0.47
"""
from __future__ import annotations

import glob
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("trafficflowbench").resolve()
DATA = ROOT / "data" / "kaggle_public"
SUB_87 = ROOT / "kaggle_research" / "semwal_087" / "output" / "submission.csv"
SUB_S24 = ROOT / "outputs" / "submissions" / "submission_s24.csv"
PROB_FILE = ROOT / "scratch" / "test_ongoing_probabilities.parquet"
KEY_FILE = DATA / "submission_key.csv"

print("Loading submission key...")
key = pd.read_csv(KEY_FILE, low_memory=False)
qu_mask = key["task"] == "queue"
key_qu = key[qu_mask].copy()

wi_files = list((DATA / "task2").glob("*/*/window_index.csv"))
wi_df = pd.concat([pd.read_csv(f) for f in wi_files], ignore_index=True)
cond_map = dict(zip(wi_df["window_id"].astype(str), wi_df["condition"].astype(str)))
origin_map = dict(zip(wi_df["window_id"].astype(str), wi_df["forecast_origin"].astype(str)))

key_qu["condition"] = key_qu["window_id"].astype(str).map(cond_map)
key_qu["forecast_origin"] = key_qu["window_id"].astype(str).map(origin_map)
key_qu["h"] = ((pd.to_datetime(key_qu["timestamp"], utc=True) - pd.to_datetime(key_qu["forecast_origin"], utc=True)).dt.total_seconds() // 300).astype(int)

# Load submissions
sub_87 = pd.read_csv(SUB_87)
sub_s24 = pd.read_csv(SUB_S24)
key_qu["pred_87"] = sub_87.loc[qu_mask, "queue_pred"].to_numpy().astype(int)
key_qu["pred_s24"] = sub_s24.loc[qu_mask, "queue_pred"].to_numpy().astype(int)

# Load test ongoing probabilities
probs = pd.read_parquet(PROB_FILE)
prob_keys = ["window_id", "h", "link_id"]
for k in prob_keys:
    probs[k] = probs[k].astype(str)
    key_qu[k] = key_qu[k].astype(str)

merged = key_qu[key_qu["condition"] == "queue_ongoing"].merge(probs[prob_keys + ["p_lgb", "p_cb", "p_s24"]], on=prob_keys, how="left")

p_s24 = merged["p_s24"].to_numpy()
p_lgb = merged["p_lgb"].to_numpy()
q_87 = merged["pred_87"].to_numpy()
q_s24 = merged["pred_s24"].to_numpy()

print(f"Loaded {len(merged):,} ongoing cells.")
print(f"0.87141 ongoing positives: {q_87.sum():,} ({q_87.mean()*100:.2f}%)")
print(f"S24 ongoing positives:     {q_s24.sum():,} ({q_s24.mean()*100:.2f}%)")

print("\n" + "=" * 75)
print(">>> EVALUATION OF ONGOING PROBABILITY THRESHOLDS & BLENDS <<<")
print("=" * 75)
print(f"{'Strategy / Model':<35} | {'Positives':<10} | {'Agree w/ 87':<12} | {'IoU w/ 87':<12}")
print("-" * 75)

for th in [0.40, 0.42, 0.44, 0.45, 0.46, 0.47, 0.48, 0.485, 0.49, 0.50]:
    pred = (p_s24 >= th).astype(int)
    inter = np.sum((pred == 1) & (q_87 == 1))
    union = np.sum((pred == 1) | (q_87 == 1))
    iou = inter / max(union, 1)
    agree = (pred == q_87).mean() * 100
    print(f"p_s24 >= {th:.3f}{'':<20} | {pred.sum():<10} | {agree:<12.2f}% | {iou:<12.4f}")

print("-" * 75)
for th in [0.42, 0.44, 0.45, 0.46, 0.48]:
    pred = (0.5 * p_s24 + 0.5 * p_lgb >= th).astype(int)
    inter = np.sum((pred == 1) & (q_87 == 1))
    union = np.sum((pred == 1) | (q_87 == 1))
    iou = inter / max(union, 1)
    agree = (pred == q_87).mean() * 100
    print(f"0.5*p_s24 + 0.5*p_lgb >= {th:.2f}{'':<10} | {pred.sum():<10} | {agree:<12.2f}% | {iou:<12.4f}")
