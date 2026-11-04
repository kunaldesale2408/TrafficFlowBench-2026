"""Generate Task 2 Precision Queue (S44):
Refines Champion S42 Queue with:
1. 2-Link High-Confidence Physical Hole Closure ([1, 0, 0, 1] where both p1, p2 >= 0.40)
2. Low-Confidence Isolated Spike Pruning ([0, 1, 0] where p < 0.25)
3. Preserves all champion S42 1-link shockwave bridges and onset calibration.

Outputs:
  outputs/submissions/queue_s44_precision.csv (174,000 rows)
"""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("D:/Kaggle/TrafficFlowBench 2026/trafficflowbench")
DATA = ROOT / "data" / "kaggle_public"
Q_S42_PATH = ROOT / "outputs" / "submissions" / "queue_s42_closure.csv"
PROB_PATH = ROOT / "scratch" / "test_ongoing_probabilities.parquet"
OUT_CSV = ROOT / "outputs" / "submissions" / "queue_s44_precision.csv"

def main():
    t0 = time.time()
    print("=" * 75)
    print(">>> GENERATING TASK 2 PRECISION QUEUE (S44) <<<")
    print("=" * 75)
    
    # 1. Load submission key
    key = pd.read_csv(DATA / "submission_key.csv", low_memory=False)
    qu_mask = key["task"] == "queue"
    key_qu = key[qu_mask].copy()
    
    # 2. Map window conditions and forecast origin
    wi_files = list((DATA / "task2").glob("*/*/window_index.csv"))
    wi_df = pd.concat([pd.read_csv(f) for f in wi_files], ignore_index=True)
    cond_map = dict(zip(wi_df["window_id"].astype(str), wi_df["condition"].astype(str)))
    origin_map = dict(zip(wi_df["window_id"].astype(str), wi_df["forecast_origin"].astype(str)))
    
    key_qu["condition"] = key_qu["window_id"].map(cond_map)
    key_qu["forecast_origin"] = key_qu["window_id"].map(origin_map)
    key_qu["h"] = ((pd.to_datetime(key_qu["timestamp"], utc=True) - pd.to_datetime(key_qu["forecast_origin"], utc=True)).dt.total_seconds() // 300).astype(int)
    
    # 3. Load S42 champion queue
    print(f"Loading champion S42 queue from {Q_S42_PATH}...")
    q_s42 = pd.read_csv(Q_S42_PATH)
    merged = key_qu.merge(q_s42, on=["window_id", "timestamp", "link_id"], how="inner")
    
    prob_keys = ["window_id", "h", "link_id"]
    probs = pd.read_parquet(PROB_PATH)
    for k in prob_keys:
        probs[k] = probs[k].astype(str)
        merged[k] = merged[k].astype(str)
    merged = merged.merge(probs[prob_keys + ["p_s24"]], on=prob_keys, how="left")
    
    initial_pos = merged["queue_pred"].sum()
    print(f"Initial S42 positive queue cells: {initial_pos:,}")
    
    panels = wi_df["panel"].unique()
    
    # 4. Bridge 2-link high confidence holes [1, 0, 0, 1] where both p1, p2 >= 0.40
    print("\nScanning for 2-link high-confidence shockwave holes [1, 0, 0, 1]...")
    bridged_2link = []
    
    for panel in panels:
        topo_path = DATA / "corridors" / panel / "network" / "lwr_mainline_topology.csv"
        if not topo_path.exists():
            continue
        topo = pd.read_csv(topo_path)
        ordered_links = topo["link_id"].astype(str).tolist()
        
        p_df = merged[(merged["panel"] == panel) & (merged["condition"] == "queue_ongoing")]
        for (w_id, h), grp in p_df.groupby(["window_id", "h"]):
            pred_map = dict(zip(grp["link_id"], grp["queue_pred"]))
            prob_map = dict(zip(grp["link_id"], grp["p_s24"]))
            idx_map = dict(zip(grp["link_id"], grp.index))
            
            arr = np.array([pred_map.get(l, 0) for l in ordered_links])
            hole2_mask = (arr[:-3] == 1) & (arr[1:-2] == 0) & (arr[2:-1] == 0) & (arr[3:] == 1)
            if hole2_mask.any():
                indices = np.where(hole2_mask)[0]
                for idx in indices:
                    l1 = ordered_links[idx + 1]
                    l2 = ordered_links[idx + 2]
                    p1 = prob_map.get(l1, 0.0)
                    p2 = prob_map.get(l2, 0.0)
                    if p1 >= 0.40 and p2 >= 0.40:
                        bridged_2link.extend([idx_map[l1], idx_map[l2]])
                        
    print(f"Bridging {len(bridged_2link)} cells across high-confidence 2-link holes.")
    merged.loc[bridged_2link, "queue_pred"] = 1
    
    # 5. Prune isolated single-link spikes [0, 1, 0] where p_s24 < 0.25
    print("\nScanning for low-confidence isolated spatial spikes [0, 1, 0]...")
    pruned_spikes = []
    
    for panel in panels:
        topo_path = DATA / "corridors" / panel / "network" / "lwr_mainline_topology.csv"
        if not topo_path.exists():
            continue
        topo = pd.read_csv(topo_path)
        ordered_links = topo["link_id"].astype(str).tolist()
        
        p_df = merged[(merged["panel"] == panel) & (merged["condition"] == "queue_ongoing")]
        for (w_id, h), grp in p_df.groupby(["window_id", "h"]):
            pred_map = dict(zip(grp["link_id"], grp["queue_pred"]))
            prob_map = dict(zip(grp["link_id"], grp["p_s24"]))
            idx_map = dict(zip(grp["link_id"], grp.index))
            
            arr = np.array([pred_map.get(l, 0) for l in ordered_links])
            spike_mask = (arr[:-2] == 0) & (arr[1:-1] == 1) & (arr[2:] == 0)
            if spike_mask.any():
                indices = np.where(spike_mask)[0] + 1
                for idx in indices:
                    link = ordered_links[idx]
                    p_val = prob_map.get(link, 0.0)
                    if p_val < 0.25:
                        pruned_spikes.append(idx_map[link])
                        
    print(f"Pruning {len(pruned_spikes)} low-confidence isolated spikes (false alarms).")
    merged.loc[pruned_spikes, "queue_pred"] = 0
    
    final_pos = merged["queue_pred"].sum()
    print(f"\nNet change: {initial_pos:,} -> {final_pos:,} positives (+{len(bridged_2link)} promoted, -{len(pruned_spikes)} cleared).")
    
    merged["split"] = np.where(merged["timestamp"].str[:7] == "2031-03", "validation (LB)", "private")
    print("\nPositives by Split:")
    print(merged.groupby("split")["queue_pred"].agg(["count", "sum"]))
    
    # 6. Save output
    out_df = merged[["window_id", "timestamp", "link_id", "queue_pred"]].copy()
    assert len(out_df) == 174_000, f"Expected 174,000 queue rows, got {len(out_df)}"
    assert set(out_df["queue_pred"].unique()).issubset({0, 1}), "Non-binary predictions!"
    assert out_df["queue_pred"].isna().sum() == 0, "NaNs in predictions!"
    
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved S44 precision queue to {OUT_CSV} in {time.time()-t0:.1f}s.")

if __name__ == "__main__":
    main()
