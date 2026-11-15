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

# Baseline: tau = 0.45
base_pred_map = dict(zip(zip(merged["window_id"], merged["h"], merged["link_id"]), (merged["prob"] >= 0.45).astype(int)))

# Evaluate baseline IoU
window_ious = []
for wid, grp in merged.groupby("window_id"):
    true = grp["queue_true"].to_numpy().astype(bool)
    pred = np.array([base_pred_map[(wid, r.h, r.link_id)] for r in grp.itertuples()])
    inter = (true & pred).sum()
    union = (true | pred).sum()
    window_ious.append(1.0 if union == 0 else inter / union)
base_iou = float(np.mean(window_ious))
print(f"Base IoU (tau=0.45): {base_iou:.5f}")

# Test adjacent shockwave extension at different tau_adj
for tau_adj in [0.42, 0.40, 0.38, 0.35, 0.32, 0.30]:
    pred_map = base_pred_map.copy()
    promoted = 0
    
    for p in panels:
        topo_path = DATA / "corridors" / p / "network" / "lwr_mainline_topology.csv"
        if not topo_path.exists(): continue
        topo = pd.read_csv(topo_path)
        ordered_links = topo['link_id'].astype(str).tolist()
        link_pos = {l: i for i, l in enumerate(ordered_links)}
        
        p_sub = merged[merged["panel"] == p]
        for (wid, h), grp in p_sub.groupby(["window_id", "h"]):
            q_set = set(l for l in grp["link_id"] if pred_map[(wid, h, l)] == 1)
            cands = [r for r in grp.itertuples() if pred_map[(wid, h, r.link_id)] == 0 and r.prob >= tau_adj]
            
            for r in cands:
                pos = link_pos.get(r.link_id)
                if pos is None: continue
                up = ordered_links[pos - 1] if pos > 0 else None
                dn = ordered_links[pos + 1] if pos < len(ordered_links) - 1 else None
                
                # Check adjacency to existing queue
                if (up in q_set) or (dn in q_set):
                    pred_map[(wid, h, r.link_id)] = 1
                    promoted += 1

    window_ious_ext = []
    for wid, grp in merged.groupby("window_id"):
        true = grp["queue_true"].to_numpy().astype(bool)
        pred = np.array([pred_map[(wid, r.h, r.link_id)] for r in grp.itertuples()])
        inter = (true & pred).sum()
        union = (true | pred).sum()
        window_ious_ext.append(1.0 if union == 0 else inter / union)
    ext_iou = float(np.mean(window_ious_ext))
    print(f"tau_adj={tau_adj:.2f} | Promoted: {promoted:3d} | Ext IoU: {ext_iou:.5f} (Delta: {ext_iou - base_iou:+.5f})")

