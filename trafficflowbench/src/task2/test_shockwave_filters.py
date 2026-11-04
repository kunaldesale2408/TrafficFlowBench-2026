"""Test Kinematic Shockwave Filters on the 80 Ground Truth Train Windows.
Measures the exact empirical IoU change of:
1. 1-link spatial hole closure ([1, 0, 1])
2. 2-link spatial hole closure ([1, 0, 0, 1])
3. Isolated spike suppression ([0, 1, 0])
"""
from pathlib import Path
import numpy as np
import pandas as pd
from eval_task2 import compute_official_task2_iou

def run_experiment():
    root = Path("trafficflowbench")
    gt = pd.read_parquet(root / "outputs" / "ground_truth" / "task2_train_ground_truth.parquet")
    print(f"Loaded ground truth: {len(gt):,} rows across {gt['window_id'].nunique()} windows.")

    # Load baseline predictions on train
    pred_path = root / "outputs" / "submissions" / "queue_train_eval.csv"
    base_pred = pd.read_csv(pred_path)
    base_pred["timestamp"] = pd.to_datetime(base_pred["timestamp"], utc=True)
    base_pred["window_id"] = base_pred["window_id"].astype(str)
    base_pred["link_id"] = base_pred["link_id"].astype(str)

    res_base = compute_official_task2_iou(base_pred, gt)
    print(f"\n[Baseline] Overall IoU: {res_base['overall_iou']:.5f} (Onset: {res_base['onset_iou']:.5f}, Ongoing: {res_base['ongoing_iou']:.5f})")

    # Add panel and horizon
    wi_files = list(Path("trafficflowbench/data/kaggle_public/task2").glob("*/train/window_index.csv"))
    wi_df = pd.concat([pd.read_csv(f) for f in wi_files], ignore_index=True)
    panel_map = dict(zip(wi_df["window_id"].astype(str), wi_df["panel"].astype(str)))
    origin_map = dict(zip(wi_df["window_id"].astype(str), wi_df["forecast_origin"].astype(str)))
    cond_map = dict(zip(wi_df["window_id"].astype(str), wi_df["condition"].astype(str)))

    base_pred["panel"] = base_pred["window_id"].map(panel_map)
    base_pred["condition"] = base_pred["window_id"].map(cond_map)
    base_pred["origin"] = pd.to_datetime(base_pred["window_id"].map(origin_map), utc=True)
    base_pred["h"] = ((base_pred["timestamp"] - base_pred["origin"]).dt.total_seconds() // 300).astype(int)

    # 1. Test 1-link hole closure on ongoing windows
    panels = base_pred["panel"].unique()
    pred_1hole = base_pred.copy()
    q_arr = pred_1hole["queue_pred"].to_numpy().copy()

    filled_1hole = 0
    for p in panels:
        topo_path = root / "data" / "kaggle_public" / "corridors" / p / "network" / "lwr_mainline_topology.csv"
        if not topo_path.exists():
            continue
        topo = pd.read_csv(topo_path, dtype=str)
        link2idx = {l: i for i, l in enumerate(topo["link_id"].tolist())}
        
        p_mask = (pred_1hole["panel"] == p) & (pred_1hole["condition"] == "queue_ongoing")
        p_indices = pred_1hole[p_mask].index.to_numpy()
        sub = pred_1hole.loc[p_indices].copy()
        sub["topo_idx"] = sub["link_id"].map(link2idx)
        sub = sub.dropna(subset=["topo_idx"]).sort_values(["window_id", "h", "topo_idx"])
        
        for (wid, h), grp in sub.groupby(["window_id", "h"]):
            idxs = grp.index.to_numpy()
            q_vals = q_arr[idxs]
            for i in range(1, len(q_vals) - 1):
                if q_vals[i-1] == 1 and q_vals[i] == 0 and q_vals[i+1] == 1:
                    q_arr[idxs[i]] = 1
                    filled_1hole += 1

    pred_1hole["queue_pred"] = q_arr
    res_1hole = compute_official_task2_iou(pred_1hole, gt)
    print(f"\n[+ 1-Link Hole Closure: {filled_1hole} bridges filled]")
    print(f"Overall IoU: {res_1hole['overall_iou']:.5f} (Onset: {res_1hole['onset_iou']:.5f}, Ongoing: {res_1hole['ongoing_iou']:.5f})")
    print(f"Delta Ongoing IoU: {res_1hole['ongoing_iou'] - res_base['ongoing_iou']:+.5f}")
    print(f"Delta Overall IoU: {res_1hole['overall_iou'] - res_base['overall_iou']:+.5f}")

    # 2. Test 2-link hole closure
    pred_2hole = pred_1hole.copy()
    q_arr2 = pred_2hole["queue_pred"].to_numpy().copy()
    filled_2hole = 0

    for p in panels:
        topo_path = root / "data" / "kaggle_public" / "corridors" / p / "network" / "lwr_mainline_topology.csv"
        if not topo_path.exists():
            continue
        topo = pd.read_csv(topo_path, dtype=str)
        link2idx = {l: i for i, l in enumerate(topo["link_id"].tolist())}
        
        p_mask = (pred_2hole["panel"] == p) & (pred_2hole["condition"] == "queue_ongoing")
        p_indices = pred_2hole[p_mask].index.to_numpy()
        sub = pred_2hole.loc[p_indices].copy()
        sub["topo_idx"] = sub["link_id"].map(link2idx)
        sub = sub.dropna(subset=["topo_idx"]).sort_values(["window_id", "h", "topo_idx"])
        
        for (wid, h), grp in sub.groupby(["window_id", "h"]):
            idxs = grp.index.to_numpy()
            q_vals = q_arr2[idxs]
            for i in range(1, len(q_vals) - 2):
                if q_vals[i-1] == 1 and q_vals[i] == 0 and q_vals[i+1] == 0 and q_vals[i+2] == 1:
                    q_arr2[idxs[i]] = 1
                    q_arr2[idxs[i+1]] = 1
                    filled_2hole += 2

    pred_2hole["queue_pred"] = q_arr2
    res_2hole = compute_official_task2_iou(pred_2hole, gt)
    print(f"\n[+ 2-Link Hole Closure: {filled_2hole} bridges filled]")
    print(f"Overall IoU: {res_2hole['overall_iou']:.5f} (Onset: {res_2hole['onset_iou']:.5f}, Ongoing: {res_2hole['ongoing_iou']:.5f})")
    print(f"Delta Ongoing IoU: {res_2hole['ongoing_iou'] - res_base['ongoing_iou']:+.5f}")
    print(f"Delta Overall IoU: {res_2hole['overall_iou'] - res_base['overall_iou']:+.5f}")

if __name__ == "__main__":
    run_experiment()
