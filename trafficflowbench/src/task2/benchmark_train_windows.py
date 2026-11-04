"""Benchmark existing models on the 80 ground-truth train windows."""
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

from eval_task2 import compute_official_task2_iou

def benchmark():
    root = Path("trafficflowbench")
    gt = pd.read_parquet(root / "outputs" / "ground_truth" / "task2_train_ground_truth.parquet")
    print(f"Loaded ground truth: {len(gt):,} rows, {gt['window_id'].nunique()} windows.")

    # 1. Evaluate persistence baseline
    # In persistence, predicted queue is simply: was the link queued at T?
    # We can read window_history for train windows
    panels = [d.name for d in (root / "data" / "kaggle_public" / "corridors").iterdir() if d.is_dir() and d.name not in {"D12_I405_N", "D12_I405_S"}]
    
    # Let's extract last known state at T from window_history
    hist_dfs = []
    for p in panels:
        hp = root / "data" / "kaggle_public" / "task2" / p / "train" / "window_history.parquet"
        if hp.exists():
            h = pd.read_parquet(hp)
            h["panel"] = p
            hist_dfs.append(h)
    
    full_hist = pd.concat(hist_dfs, ignore_index=True)
    full_hist["timestamp"] = pd.to_datetime(full_hist["timestamp"], utc=True)
    
    # For each window and link, find latest speed in history
    latest_speeds = full_hist.sort_values("timestamp").groupby(["window_id", "link_id"]).last().reset_index()
    
    # Load v_cut map
    v_cut_map = {}
    for p in panels:
        fd = pd.read_csv(root / "data" / "kaggle_public" / "corridors" / p / "network" / "fd_parameters.csv").drop_duplicates("link_id")
        v_cut_col = "v_cut" if "v_cut" in fd.columns else None
        for r in fd.itertuples():
            v_cut_map[str(r.link_id)] = float(getattr(r, v_cut_col)) if v_cut_col else 0.6 * float(r.free_speed_kmh)
            
    latest_speeds["v_cut"] = latest_speeds["link_id"].astype(str).map(v_cut_map)
    latest_speeds["persist_queue"] = ((latest_speeds["speed_kmh"] <= latest_speeds["v_cut"]) & latest_speeds["is_score_eligible"].astype(bool)).astype(int)
    
    # Merge persistence to all target timesteps
    persist_pred = gt[["window_id", "timestamp", "link_id"]].merge(
        latest_speeds[["window_id", "link_id", "persist_queue"]], on=["window_id", "link_id"], how="left"
    ).fillna(0)
    persist_pred["queue_pred"] = persist_pred["persist_queue"].astype(int)
    
    res_persist = compute_official_task2_iou(persist_pred, gt)
    print("\n--- Persistence Baseline ---")
    print(f"Overall IoU: {res_persist['overall_iou']:.5f} (Onset: {res_persist['onset_iou']:.5f}, Ongoing: {res_persist['ongoing_iou']:.5f})")

if __name__ == "__main__":
    benchmark()
