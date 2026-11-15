"""Diagnostic script: Inspect LWR conservation terms in detail.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path("D:/Kaggle/TrafficFlowBench 2026/trafficflowbench")
sys.path.insert(0, str(ROOT / "official_toolkit" / "src"))

from task3.score_task3 import (
    network_parameters,
    link_order,
    load_ramp_flows,
    topology_boundary_flows,
    state_for_regime,
    derive_physics,
)

def main():
    release = ROOT / "data" / "kaggle_public"
    panel = "D12_I5_N"
    split = "validation"
    regime = "R1"
    
    state_path = ROOT / "kaggle_research" / "semwal_087" / "output" / "task1" / "task1_validation.csv"
    print(f"Loading {state_path}...")
    state_df = pd.read_csv(state_path)
    state_df["panel"] = state_df["panel"].astype(str)
    state_df["link_id"] = state_df["link_id"].astype(str)
    state_df["mask_regime"] = state_df["mask_regime"].astype(str)
    state_df["timestamp"] = pd.to_datetime(state_df["timestamp"], utc=True)
    
    state_panel = state_df[state_df["panel"] == panel]
    
    # Derive physics frame
    phys_df, missing = derive_physics(
        panel=panel,
        release=release,
        split=split,
        state_panel=state_panel,
        regime=regime,
    )
    print(f"Derived physics frame: {len(phys_df):,} rows.")
    
    # Now replicate the score_panel logic on phys_df
    p = network_parameters(release / "corridors" / panel).set_index("link_id")
    x = phys_df.join(p, on="link_id", rsuffix="_net")
    
    topo_path = release / "corridors" / panel / "network" / "lwr_mainline_topology.csv"
    topo = pd.read_csv(topo_path, dtype=str)
    topo_ids = set(topo.link_id.astype(str))
    submitted_ids = set(x.link_id.astype(str))
    observable_by_link = {}
    for row in topo.itertuples(index=False):
        incoming = {v.strip() for v in str(getattr(row, "incoming_link_ids", "")).split(";") if v.strip()}
        outgoing = {v.strip() for v in str(getattr(row, "outgoing_link_ids", "")).split(";") if v.strip()}
        observable_by_link[str(row.link_id)] = (
            (incoming & topo_ids).issubset(submitted_ids)
            and (outgoing & topo_ids).issubset(submitted_ids)
        )
    x = x[x.link_id.astype(str).map(observable_by_link).fillna(False)].copy()
    x = x.sort_values(["link_id", "timestamp"])
    
    x["dN"] = x.groupby("link_id").accumulation_N.shift(-1) - x.accumulation_N
    x["dt"] = (x.groupby("link_id").timestamp.shift(-1) - x.timestamp).dt.total_seconds() / 3600.0
    x = x[(x.dt > 0) & (x.dt <= 5.1 / 60.0)].copy()
    
    # Filter ramps valid
    x = x[(x.on_ramp_valid >= 0.5) & (x.off_ramp_valid >= 0.5)].copy()
    
    rhs = x.dt * (x.inflow_vph + x.on_ramp_flow_vph - x.outflow_vph - x.off_ramp_flow_vph)
    resid = (x.dN - rhs).abs()
    
    print("\n=== LWR DIAGNOSTICS ===")
    print(f"Total valid transitions: {len(x):,}")
    print(f"Sum |RHS|: {rhs.abs().sum():,.2f}")
    print(f"Sum |dN|:  {x.dN.abs().sum():,.2f}")
    print(f"Sum |dN - RHS| (Residual): {resid.sum():,.2f}")
    print(f"E_LWR = Residual / Sum|RHS| = {resid.sum() / (rhs.abs().sum() + 1e-9):.4f}")
    print(f"Mean RHS: {rhs.mean():.4f}, Std RHS: {rhs.std():.4f}")
    print(f"Mean dN:  {x.dN.mean():.4f}, Std dN:  {x.dN.std():.4f}")
    print(f"Correlation between dN and RHS: {x.dN.corr(rhs):.4f}")
    
    # Check largest RHS rows
    x["rhs"] = rhs
    x["abs_rhs"] = rhs.abs()
    x["abs_resid"] = resid
    top_resid = x.sort_values("abs_resid", ascending=False).head(10)
    print("\n=== TOP 10 LARGEST RESIDUAL ROWS ===")
    cols_debug = ["timestamp", "link_id", "speed_kmh", "flow_vph", "inflow_vph", "on_ramp_flow_vph", "outflow_vph", "off_ramp_flow_vph", "dN", "rhs", "abs_resid"]
    print(top_resid[cols_debug].to_string())
    
    # Inspect links with lowest residual
    print("\n=== TOP 10 SMALLEST RESIDUAL ROWS ===")
    print(x.sort_values("abs_resid").head(10)[cols_debug].to_string())

if __name__ == "__main__":
    main()
