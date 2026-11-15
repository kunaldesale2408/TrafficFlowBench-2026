import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, ".")
sys.path.insert(0, "trafficflowbench")

ROOT = Path("trafficflowbench")
DATA = ROOT / "data" / "kaggle_public"

from trafficflowbench.scratch.evaluate_queue_precision_tuning import merged

panels = merged["panel"].unique()

# Let's test topological hole closure at tau = 0.35 and tau = 0.30
for test_tau in [0.45, 0.40, 0.35, 0.30, 0.29]:
    pred_map = dict(zip(zip(merged["window_id"], merged["h"], merged["link_id"]), (merged["prob"] >= test_tau).astype(int)))
    
    # Calculate baseline IoU at this tau
    window_ious = []
    for wid, grp in merged.groupby("window_id"):
        true = grp["queue_true"].to_numpy().astype(bool)
        pred = np.array([pred_map[(wid, r.h, r.link_id)] for r in grp.itertuples()])
        inter = (true & pred).sum()
        union = (true | pred).sum()
        window_ious.append(1.0 if union == 0 else inter / union)
    base_iou = float(np.mean(window_ious))
    
    # Now apply topological hole closure
    # 1. 1-link hole closure: [1, 0, 1] where p >= 0.25
    filled_1 = 0
    filled_2 = 0
    pruned_spikes = 0
    
    for p in panels:
        topo_path = DATA / "corridors" / p / "network" / "lwr_mainline_topology.csv"
        if not topo_path.exists(): continue
        topo = pd.read_csv(topo_path)
        ordered_links = topo["link_id"].astype(str).tolist()
        
        p_sub = merged[merged["panel"] == p]
        for (wid, h), grp in p_sub.groupby(["window_id", "h"]):
            prob_dict = dict(zip(grp["link_id"], grp["prob"]))
            arr = np.array([pred_map.get((wid, h, l), 0) for l in ordered_links])
            
            # 1-link hole
            hole1 = (arr[:-2] == 1) & (arr[1:-1] == 0) & (arr[2:] == 1)
            for idx in np.where(hole1)[0]:
                mid_link = ordered_links[idx + 1]
                if prob_dict.get(mid_link, 0.0) >= 0.25:
                    pred_map[(wid, h, mid_link)] = 1
                    filled_1 += 1
                    
            # 2-link hole
            arr = np.array([pred_map.get((wid, h, l), 0) for l in ordered_links])
            hole2 = (arr[:-3] == 1) & (arr[1:-2] == 0) & (arr[2:-1] == 0) & (arr[3:] == 1)
            for idx in np.where(hole2)[0]:
                l1 = ordered_links[idx + 1]
                l2 = ordered_links[idx + 2]
                if prob_dict.get(l1, 0.0) >= 0.25 and prob_dict.get(l2, 0.0) >= 0.25:
                    pred_map[(wid, h, l1)] = 1
                    pred_map[(wid, h, l2)] = 1
                    filled_2 += 2
                    
            # Spike pruning: [0, 1, 0] where prob < 0.15
            arr = np.array([pred_map.get((wid, h, l), 0) for l in ordered_links])
            spikes = (arr[:-2] == 0) & (arr[1:-1] == 1) & (arr[2:] == 0)
            for idx in np.where(spikes)[0]:
                mid_link = ordered_links[idx + 1]
                if prob_dict.get(mid_link, 0.0) < 0.15:
                    pred_map[(wid, h, mid_link)] = 0
                    pruned_spikes += 1

    # Calculate filtered IoU
    window_ious_filt = []
    for wid, grp in merged.groupby("window_id"):
        true = grp["queue_true"].to_numpy().astype(bool)
        pred = np.array([pred_map[(wid, r.h, r.link_id)] for r in grp.itertuples()])
        inter = (true & pred).sum()
        union = (true | pred).sum()
        window_ious_filt.append(1.0 if union == 0 else inter / union)
    filt_iou = float(np.mean(window_ious_filt))
    
    print(f"tau={test_tau:.2f} | Base IoU: {base_iou:.5f} -> Filtered IoU: {filt_iou:.5f} (Delta: {filt_iou - base_iou:+.5f}, Filled: {filled_1 + filled_2}, Pruned: {pruned_spikes})")

