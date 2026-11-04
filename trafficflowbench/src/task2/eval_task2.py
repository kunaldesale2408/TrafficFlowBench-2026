"""Official IoU Evaluator for Task 2 using Ground Truth."""
from pathlib import Path
import numpy as np
import pandas as pd

def compute_official_task2_iou(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> dict:
    """Computes exact competition IoU:
    1. For each window, IoU = (pred & true).sum() / max((pred | true).sum(), 1) on eligible cells.
    2. Group by (panel, condition) -> mean IoU.
    3. For each panel, condition_mean = (onset_mean + ongoing_mean) / 2.
    4. Overall IoU = mean of panel condition_means across all panels.
    """
    keys = ["window_id", "timestamp", "link_id"]
    
    gt_df = gt_df.copy()
    gt_df["timestamp"] = pd.to_datetime(gt_df["timestamp"], utc=True)
    gt_df["window_id"] = gt_df["window_id"].astype(str)
    gt_df["link_id"] = gt_df["link_id"].astype(str)
    
    pred_df = pred_df.copy()
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"], utc=True)
    pred_df["window_id"] = pred_df["window_id"].astype(str)
    pred_df["link_id"] = pred_df["link_id"].astype(str)
    
    # Merge on keys
    merged = gt_df[gt_df["is_score_eligible"].astype(bool)].merge(
        pred_df[keys + ["queue_pred"]], on=keys, how="inner"
    )
    
    window_ious = []
    for wid, grp in merged.groupby("window_id"):
        true = grp["queue_true"].astype(bool).to_numpy()
        pred = grp["queue_pred"].astype(bool).to_numpy()
        inter = (true & pred).sum()
        union = (true | pred).sum()
        iou = 1.0 if union == 0 else (inter / union)
        panel = grp["panel"].iloc[0]
        cond = grp["condition"].iloc[0]
        window_ious.append({
            "window_id": wid,
            "panel": panel,
            "condition": cond,
            "iou": float(iou),
            "inter": int(inter),
            "union": int(union),
            "n_true": int(true.sum()),
            "n_pred": int(pred.sum()),
        })
    
    w_df = pd.DataFrame(window_ious)
    
    # Group by panel and condition
    pc_mean = w_df.groupby(["panel", "condition"])["iou"].mean().unstack()
    pc_mean["panel_mean"] = pc_mean.mean(axis=1)
    
    overall_iou = float(pc_mean["panel_mean"].mean())
    onset_iou = float(pc_mean["queue_onset"].mean()) if "queue_onset" in pc_mean else 0.0
    ongoing_iou = float(pc_mean["queue_ongoing"].mean()) if "queue_ongoing" in pc_mean else 0.0
    
    return {
        "overall_iou": overall_iou,
        "onset_iou": onset_iou,
        "ongoing_iou": ongoing_iou,
        "by_panel": pc_mean.to_dict(),
        "window_df": w_df,
    }

if __name__ == "__main__":
    gt = pd.read_parquet("trafficflowbench/outputs/ground_truth/task2_train_ground_truth.parquet")
    # Quick sanity check: perfect prediction
    perfect = gt[["window_id", "timestamp", "link_id", "queue_true"]].rename(columns={"queue_true": "queue_pred"})
    res = compute_official_task2_iou(perfect, gt)
    print(f"Sanity Check (Perfect): Overall IoU = {res['overall_iou']:.5f}")
    assert abs(res["overall_iou"] - 1.0) < 1e-6
