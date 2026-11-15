"""Analyze 2D Space-Time Holes & Isolated Spikes in Champion Queue (S42).
Inspects both spatial [1, 0, 1] along mileposts and temporal [1, 0, 1] across horizon steps.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("D:/Kaggle/TrafficFlowBench 2026/trafficflowbench")
DATA = ROOT / "data" / "kaggle_public"
Q_S42_PATH = ROOT / "outputs" / "submissions" / "queue_s42_closure.csv"
PROB_PATH = ROOT / "scratch" / "test_ongoing_probabilities.parquet"

def main():
    print("=" * 70)
    print("Scanning Champion S42 Queue for 2D Spatiotemporal Continuity Patterns")
    print("=" * 70)
    
    key = pd.read_csv(DATA / "submission_key.csv", low_memory=False)
    q_df = pd.read_csv(Q_S42_PATH)
    probs = pd.read_parquet(PROB_PATH)
    
    # Merge metadata
    qu_key = key[key["task"] == "queue"].copy()
    qu_key["window_id"] = qu_key["window_id"].astype(str)
    qu_key["timestamp"] = qu_key["timestamp"].astype(str)
    qu_key["link_id"] = qu_key["link_id"].astype(str)
    
    wi_files = list((DATA / "task2").glob("*/*/window_index.csv"))
    wi_df = pd.concat([pd.read_csv(f) for f in wi_files], ignore_index=True)
    cond_map = dict(zip(wi_df["window_id"].astype(str), wi_df["condition"].astype(str)))
    origin_map = dict(zip(wi_df["window_id"].astype(str), wi_df["forecast_origin"].astype(str)))
    
    qu_key["condition"] = qu_key["window_id"].map(cond_map)
    qu_key["forecast_origin"] = qu_key["window_id"].map(origin_map)
    qu_key["h"] = ((pd.to_datetime(qu_key["timestamp"], utc=True) - pd.to_datetime(qu_key["forecast_origin"], utc=True)).dt.total_seconds() // 300).astype(int)
    
    m = qu_key.merge(q_df, on=["window_id", "timestamp", "link_id"], how="inner")
    
    prob_keys = ["window_id", "h", "link_id"]
    for k in prob_keys:
        probs[k] = probs[k].astype(str)
        m[k] = m[k].astype(str)
    m = m.merge(probs[prob_keys + ["p_s24"]], on=prob_keys, how="left")
    
    print(f"Total merged queue records: {len(m):,}")
    print(f"Total S42 positive queue cells: {m['queue_pred'].sum():,}")
    
    # 1. Temporal [1, 0, 1] Holes for Ongoing Windows
    # For a fixed (window_id, link_id), across h = 1..6:
    temp_holes = []
    ongoing_m = m[m["condition"] == "queue_ongoing"].copy()
    ongoing_m["h"] = ongoing_m["h"].astype(int)
    
    piv_time = ongoing_m.pivot_table(index=["window_id", "link_id"], columns="h", values="queue_pred")
    piv_prob = ongoing_m.pivot_table(index=["window_id", "link_id"], columns="h", values="p_s24")
    
    # Detect pattern [1, 0, 1] across h
    # Possible centers: h=2 (1,2,3), h=3 (2,3,4), h=4 (3,4,5), h=5 (4,5,6)
    for c in [2, 3, 4, 5]:
        if (c-1 in piv_time.columns) and (c in piv_time.columns) and (c+1 in piv_time.columns):
            hole = (piv_time[c-1] == 1) & (piv_time[c] == 0) & (piv_time[c+1] == 1)
            if hole.any():
                for (w, l) in piv_time[hole].index:
                    p_val = piv_prob.loc[(w, l), c]
                    temp_holes.append({"window_id": w, "h": c, "link_id": l, "p_val": p_val})
                
    temp_df = pd.DataFrame(temp_holes)
    print(f"\n1. Temporal Dropout Holes [1, 0, 1] across time: {len(temp_df)}")
    if len(temp_df) > 0:
        print("Distribution of continuous p_s24 at temporal holes:")
        print(temp_df["p_val"].describe())
        print(f"Temporal holes with p_s24 >= 0.30: {(temp_df['p_val'] >= 0.30).sum()}")
        print(f"Temporal holes with p_s24 >= 0.40: {(temp_df['p_val'] >= 0.40).sum()}")
        
    # 2. Isolated Spikes [0, 1, 0] along space (false alarms)
    # Check spatial continuity along ordered links for each window and horizon
    isolated_spikes = []
    panels = wi_df["panel"].unique()
    
    for panel in panels:
        topo_path = DATA / "corridors" / panel / "network" / "lwr_mainline_topology.csv"
        if not topo_path.exists():
            continue
        topo = pd.read_csv(topo_path)
        ordered_links = topo["link_id"].astype(str).tolist()
        
        p_df = m[(m["panel"] == panel) & (m["condition"] == "queue_ongoing")]
        for (w_id, h), grp in p_df.groupby(["window_id", "h"]):
            pred_map = dict(zip(grp["link_id"], grp["queue_pred"]))
            prob_map = dict(zip(grp["link_id"], grp["p_s24"]))
            
            arr = np.array([pred_map.get(l, 0) for l in ordered_links])
            spike_mask = (arr[:-2] == 0) & (arr[1:-1] == 1) & (arr[2:] == 0)
            if spike_mask.any():
                indices = np.where(spike_mask)[0] + 1
                for idx in indices:
                    link = ordered_links[idx]
                    p_val = prob_map.get(link, 0.0)
                    isolated_spikes.append({"window_id": w_id, "h": h, "link_id": link, "p_val": p_val, "panel": panel})
                    
    spike_df = pd.DataFrame(isolated_spikes)
    # 3. Spatial 2-Link Holes [1, 0, 0, 1]
    two_link_holes = []
    for panel in panels:
        topo_path = DATA / "corridors" / panel / "network" / "lwr_mainline_topology.csv"
        if not topo_path.exists():
            continue
        topo = pd.read_csv(topo_path)
        ordered_links = topo["link_id"].astype(str).tolist()
        
        p_df = m[(m["panel"] == panel) & (m["condition"] == "queue_ongoing")]
        for (w_id, h), grp in p_df.groupby(["window_id", "h"]):
            pred_map = dict(zip(grp["link_id"], grp["queue_pred"]))
            prob_map = dict(zip(grp["link_id"], grp["p_s24"]))
            
            arr = np.array([pred_map.get(l, 0) for l in ordered_links])
            hole2_mask = (arr[:-3] == 1) & (arr[1:-2] == 0) & (arr[2:-1] == 0) & (arr[3:] == 1)
            if hole2_mask.any():
                indices = np.where(hole2_mask)[0]
                for idx in indices:
                    l1 = ordered_links[idx + 1]
                    l2 = ordered_links[idx + 2]
                    p1 = prob_map.get(l1, 0.0)
                    p2 = prob_map.get(l2, 0.0)
                    two_link_holes.append({"window_id": w_id, "h": h, "link1": l1, "link2": l2, "p1": p1, "p2": p2, "panel": panel})
                    
    hole2_df = pd.DataFrame(two_link_holes)
    print(f"\n3. Spatial 2-Link Holes [1, 0, 0, 1]: {len(hole2_df)}")
    if len(hole2_df) > 0:
        both_high = (hole2_df["p1"] >= 0.35) & (hole2_df["p2"] >= 0.35)
        print(f"2-link holes where BOTH p1, p2 >= 0.35: {both_high.sum()}")
        both_higher = (hole2_df["p1"] >= 0.40) & (hole2_df["p2"] >= 0.40)
        print(f"2-link holes where BOTH p1, p2 >= 0.40: {both_higher.sum()}")
        print(hole2_df[both_high][["window_id", "h", "link1", "link2", "p1", "p2"]].head(10).to_string())

if __name__ == "__main__":
    main()
