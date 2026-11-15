"""Test Flux Conservation along Highway Corridors:
Compare raw ML predicted flows against topological upstream/downstream continuity.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("D:/Kaggle/TrafficFlowBench 2026/trafficflowbench")
DATA = ROOT / "data" / "kaggle_public"

def test_panel(panel="D12_I5_N", split="validation", regime="R1"):
    print("=" * 70)
    print(f"Testing Flux Conservation on {panel} [{split}, {regime}]")
    print("=" * 70)
    
    panel_dir = DATA / "corridors" / panel
    topo_path = panel_dir / "network" / "lwr_mainline_topology.csv"
    topo = pd.read_csv(topo_path, dtype=str)
    topo["link_id"] = topo["link_id"].astype(str)
    
    # Load state predictions
    state_path = ROOT / "kaggle_research" / "semwal_087" / "output" / "task1" / "task1_validation.csv"
    state_df = pd.read_csv(state_path)
    state_df = state_df[(state_df["panel"] == panel) & (state_df["mask_regime"] == regime)]
    state_df["timestamp"] = pd.to_datetime(state_df["timestamp"], utc=True)
    state_df["link_id"] = state_df["link_id"].astype(str)
    
    print(f"Loaded {len(state_df):,} state target cells.")
    
    # Check linear 1-to-1 links without ramps
    simple_links = topo[
        (topo["n_incoming"] == 1) & 
        (topo["n_outgoing"] == 1) & 
        (topo["on_ramp_link_ids"].isna()) & 
        (topo["off_ramp_link_ids"].isna())
    ]
    print(f"Identified {len(simple_links)} simple 1-to-1 no-ramp mainline links.")
    
    # Map upstream link
    up_map = dict(zip(simple_links["link_id"], simple_links["incoming_link_ids"]))
    down_map = dict(zip(simple_links["link_id"], simple_links["outgoing_link_ids"]))
    
    # For a sample day/hour, compare link flow to upstream and downstream flow
    timestamps = sorted(state_df["timestamp"].unique())
    print(f"Total timestamps: {len(timestamps)}")
    
    piv = state_df.pivot_table(index="timestamp", columns="link_id", values="flow_vph")
    
    # Compute continuity discrepancy
    discrepancies = []
    for link, up_link in up_map.items():
        if link in piv.columns and up_link in piv.columns:
            s_link = piv[link]
            s_up = piv[up_link]
            valid = s_link.notna() & s_up.notna()
            if valid.sum() > 0:
                diff = (s_link[valid] - s_up[valid]).abs()
                discrepancies.append(diff.mean())
                
    if discrepancies:
        print(f"Average adjacent link flow discrepancy on masked cells: {np.mean(discrepancies):.2f} vph")
        print(f"Median discrepancy: {np.median(discrepancies):.2f} vph")
        print(f"Max discrepancy: {np.max(discrepancies):.2f} vph")
    else:
        print("No simultaneously masked adjacent pairs.")

if __name__ == "__main__":
    test_panel()
