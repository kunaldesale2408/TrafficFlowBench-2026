"""Data loader utility for TrafficFlowBench.

Reads corridor mainline states (masked/unmasked), ramp flows, and configuration
from the kaggle_public dataset release.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def get_corridor_list(release_root: Union[str, Path]) -> List[str]:
    """Return the list of 10 corridor panel IDs from corridors.json."""
    root = Path(release_root)
    corridors_json = root / "config" / "corridors.json"
    if corridors_json.exists():
        data = json.loads(corridors_json.read_text(encoding="utf-8"))
        return [p["corridor_id"] for p in data.get("panels", [])]
    return [
        "D7_I10_E", "D7_I10_W",
        "D7_I210_E", "D7_I210_W",
        "D7_I405_N", "D7_I405_S",
        "D12_I5_N", "D12_I5_S",
        "D12_I405_N", "D12_I405_S",
    ]


def load_corridor_config(release_root: Union[str, Path], panel: str) -> Dict[str, Any]:
    """Load network topology, links, and fundamental diagram parameters for a corridor."""
    root = Path(release_root)
    panel_dir = root / "corridors" / panel
    network_dir = panel_dir / "network"

    config: Dict[str, Any] = {"panel": panel, "panel_dir": panel_dir}

    links_path = network_dir / "links.csv"
    if links_path.exists():
        links_df = pd.read_csv(links_path, dtype={"link_id": str})
        config["links"] = links_df

    fd_path = network_dir / "fd_parameters.csv"
    if fd_path.exists():
        fd_df = pd.read_csv(fd_path, dtype={"link_id": str})
        config["fd_parameters"] = fd_df

    ramp_map_path = network_dir / "ramp_attachment_map.csv"
    if ramp_map_path.exists():
        ramp_map_df = pd.read_csv(ramp_map_path, dtype={"ramp_link_id": str, "ramp_type": str, "nearest_mainline_link_id": str})
        config["ramp_attachment_map"] = ramp_map_df

    topology_path = network_dir / "lwr_mainline_topology.csv"
    if topology_path.exists():
        topo_df = pd.read_csv(topology_path, dtype={"upstream_link_id": str, "downstream_link_id": str})
        config["topology"] = topo_df

    return config


def load_corridor(
    release_root: Union[str, Path],
    panel: str,
    split: str = "train",
    masked: bool = True,
    include_ramps: bool = True,
    sample_n_files: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load a single corridor's mainline states, ramp states, and config.

    CRITICAL RULE: Null or missing ramp flows are represented as NaN, NEVER as 0.
    Treating missing ramp flow as 0 will break conservation and destroy Task 3 scoring.

    Parameters:
    -----------
    release_root: Path to the root of the unpacked kaggle_public directory.
    panel: Corridor panel name (e.g. 'D12_I5_N').
    split: 'train', 'validation', or 'private'.
    masked: If True, loads mainline_states_masked; else loads mainline_states (train only).
    include_ramps: If True, joins on-ramp and off-ramp flows onto mainline dataframe.
    sample_n_files: If set, loads only the first N parquet files (useful for fast testing).

    Returns:
    --------
    mainline_df: pandas DataFrame containing mainline states and optional ramp flows.
    config: dict containing corridor metadata, links table, and FD parameters.
    """
    root = Path(release_root)
    panel_dir = root / "corridors" / panel
    if not panel_dir.exists():
        raise FileNotFoundError(f"Corridor directory not found: {panel_dir}")

    config = load_corridor_config(release_root, panel)

    # Mainline files
    mainline_sub = "mainline_states_masked" if masked or split != "train" else "mainline_states"
    mainline_dir = panel_dir / split / mainline_sub
    if not mainline_dir.exists() and split == "train" and not masked:
        mainline_dir = panel_dir / split / "mainline_states_masked"

    parquet_files = sorted(mainline_dir.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {mainline_dir}")

    if sample_n_files is not None and sample_n_files > 0:
        parquet_files = parquet_files[:sample_n_files]

    chunks = []
    for p in parquet_files:
        chunks.append(pd.read_parquet(p))
    df = pd.concat(chunks, ignore_index=True)

    # Cast string identifiers
    df["station_id"] = df["station_id"].astype(str)
    df["link_id"] = df["link_id"].astype(str)
    if "panel" not in df.columns:
        df["panel"] = panel

    # Load ramp states if requested
    if include_ramps:
        ramp_dir = panel_dir / split / "ramp_states"
        if ramp_dir.exists():
            ramp_files = sorted(ramp_dir.glob("**/*.parquet"))
            if sample_n_files is not None and sample_n_files > 0:
                ramp_files = ramp_files[:sample_n_files]

            if ramp_files:
                ramp_chunks = [pd.read_parquet(rf) for rf in ramp_files]
                ramp_df = pd.concat(ramp_chunks, ignore_index=True)
                ramp_df["station_id"] = ramp_df["station_id"].astype(str)
                ramp_df["flow_vph"] = pd.to_numeric(ramp_df["flow_vph"], errors="coerce")

                # Attach ramp attachment map if available
                ramp_map = config.get("ramp_attachment_map")
                if ramp_map is not None:
                    ramp_map_clean = ramp_map[["ramp_link_id", "ramp_type", "nearest_mainline_link_id"]].drop_duplicates("ramp_link_id")
                    ramp_df = ramp_df.merge(ramp_map_clean, on="ramp_link_id", how="left", suffixes=("", "_map"))
                    
                    target_link = ramp_df["nearest_mainline_link_id"].astype(str)
                    r_type = ramp_df["ramp_type"]
                    on_mask = r_type.isin(["on", "OR", "ON"])
                    off_mask = r_type.isin(["off", "FR", "OFF"])
                    
                    ramp_df["target_link_id"] = target_link
                    on_ramps = ramp_df[on_mask].groupby(
                        ["timestamp", "target_link_id"], as_index=False
                    )["flow_vph"].sum().rename(columns={"target_link_id": "link_id", "flow_vph": "on_ramp_flow_vph"})

                    off_ramps = ramp_df[off_mask].groupby(
                        ["timestamp", "target_link_id"], as_index=False
                    )["flow_vph"].sum().rename(columns={"target_link_id": "link_id", "flow_vph": "off_ramp_flow_vph"})

                    df = df.merge(on_ramps, on=["timestamp", "link_id"], how="left")
                    df = df.merge(off_ramps, on=["timestamp", "link_id"], how="left")
                    # NOTE: Do NOT fillna(0) for ramp flows! Keep as NaN when unobserved!

    return df, config


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test data loader")
    parser.add_argument("--release-root", type=str, default="./data/kaggle_public")
    parser.add_argument("--panel", type=str, default="D12_I5_N")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    try:
        df, cfg = load_corridor(args.release_root, args.panel, split=args.split, sample_n_files=1)
        print(f"Successfully loaded {len(df)} rows for panel {args.panel} ({args.split}).")
        print(f"Columns: {list(df.columns)}")
        if "on_ramp_flow_vph" in df.columns:
            print(f"On-ramp flow nulls: {df['on_ramp_flow_vph'].isna().sum()} (preserved as NaN)")
    except Exception as e:
        print(f"Could not load data: {e}")
        print("Paste your Kaggle competition data into data/kaggle_public/ and re-run.")
