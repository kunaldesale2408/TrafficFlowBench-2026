"""Analyze and Tune Horizon-Decaying Threshold Schedule for Task 2.
Compares S44 predictions against true empirical queue decay rates across h = 1..6.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("D:/Kaggle/TrafficFlowBench 2026/trafficflowbench")
DATA = ROOT / "data" / "kaggle_public"
GT_PATH = ROOT / "outputs" / "ground_truth" / "task2_train_ground_truth.parquet"
PROB_PATH = ROOT / "scratch" / "test_ongoing_probabilities.parquet"
S44_PATH = ROOT / "outputs" / "submissions" / "queue_s44_precision.csv"

def main():
    print("=" * 70)
    print("Tuning Horizon-Decaying Ongoing Queue Thresholds")
    print("=" * 70)
    
    # 1. Ground truth ongoing queue distribution across h = 1..6
    gt = pd.read_parquet(GT_PATH)
    gt_ongoing = gt[gt["condition"] == "queue_ongoing"].copy()
    gt_ongoing["h"] = gt_ongoing.groupby(["window_id", "link_id"])["timestamp"].rank().astype(int)
    
    gt_stats = gt_ongoing.groupby("h")["queue_true"].agg(["count", "sum", "mean"])
    print("\n1. Ground Truth Ongoing Queue Rate by Horizon (Train Windows):")
    for h, row in gt_stats.iterrows():
        print(f"  h={h} (T+{h*5:02d}m): {int(row['sum']):,} queues / {int(row['count']):,} links = {row['mean']:.4%}")
    
    # 2. S44 Ongoing Queue Distribution
    s44 = pd.read_csv(S44_PATH)
    wi_files = list((DATA / "task2").glob("*/*/window_index.csv"))
    wi_df = pd.concat([pd.read_csv(f) for f in wi_files], ignore_index=True)
    cond_map = dict(zip(wi_df["window_id"].astype(str), wi_df["condition"].astype(str)))
    origin_map = dict(zip(wi_df["window_id"].astype(str), wi_df["forecast_origin"].astype(str)))
    
    s44["condition"] = s44["window_id"].map(cond_map)
    s44["forecast_origin"] = s44["window_id"].map(origin_map)
    s44["h"] = ((pd.to_datetime(s44["timestamp"], utc=True) - pd.to_datetime(s44["forecast_origin"], utc=True)).dt.total_seconds() // 300).astype(int)
    
    s44_ongoing = s44[s44["condition"] == "queue_ongoing"]
    s44_stats = s44_ongoing.groupby("h")["queue_pred"].agg(["count", "sum", "mean"])
    
    print("\n2. S44 Champion Ongoing Queue Rate by Horizon (Test Windows):")
    for h, row in s44_stats.iterrows():
        print(f"  h={h} (T+{h*5:02d}m): {int(row['sum']):,} queues / {int(row['count']):,} links = {row['mean']:.4%}")
        
    # 3. Model Probabilities across h in test_ongoing_probabilities
    probs = pd.read_parquet(PROB_PATH)
    probs_ongoing = probs.merge(wi_df[["window_id", "condition"]].drop_duplicates(), on="window_id")
    probs_ongoing = probs_ongoing[probs_ongoing["condition"] == "queue_ongoing"]
    
    print("\n3. Continuous Model Probability (p_s24) Quantiles across Horizon Steps:")
    for h in range(1, 7):
        p_h = probs_ongoing[probs_ongoing["h"].astype(int) == h]["p_s24"]
        print(f"  h={h}: mean={p_h.mean():.4f}, median={p_h.median():.4f}, p85={p_h.quantile(0.85):.4f}, p90={p_h.quantile(0.90):.4f}, p95={p_h.quantile(0.95):.4f}")

if __name__ == "__main__":
    main()
