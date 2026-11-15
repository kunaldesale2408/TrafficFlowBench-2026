import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("trafficflowbench")
gt = pd.read_parquet(ROOT / "outputs" / "ground_truth" / "task2_train_ground_truth.parquet")

ong_gt = gt[gt["condition"] == "queue_ongoing"].copy()

# Add horizon step h (1 to 6)
# Each window has 6 timestamps
ong_gt = ong_gt.sort_values(["window_id", "timestamp", "link_id"])
time_map = {t: i+1 for i, t in enumerate(sorted(ong_gt["timestamp"].unique()))}
# Actually group by window_id to get h
ong_gt["h"] = ong_gt.groupby("window_id")["timestamp"].rank(method="dense").astype(int)

print("--- QUEUE POSITIVE COUNT BY HORIZON (40 TRAIN ONGOING WINDOWS) ---")
h_stats = ong_gt.groupby("h")["queue_true"].agg(["count", "sum", "mean"])
print(h_stats)

print("\n--- LINK TRANSITIONS FROM h TO h+1 ---")
# Pivot to (window_id, link_id) x h
piv = ong_gt.pivot(index=["window_id", "link_id"], columns="h", values="queue_true")
for h in range(1, 6):
    q_curr = piv[h]
    q_next = piv[h+1]
    stay_queued = ((q_curr == 1) & (q_next == 1)).sum()
    cleared = ((q_curr == 1) & (q_next == 0)).sum()
    newly_queued = ((q_curr == 0) & (q_next == 1)).sum()
    stay_free = ((q_curr == 0) & (q_next == 0)).sum()
    persist_iou = stay_queued / (stay_queued + cleared + newly_queued)
    print(f"h={h} -> h={h+1}: Stay={stay_queued}, Cleared={cleared}, New={newly_queued}, IoU={persist_iou:.4f}")

