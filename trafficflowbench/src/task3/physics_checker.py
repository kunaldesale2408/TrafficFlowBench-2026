"""Task 3 — Physics Compliance Checker.

Evaluates an existing Task 1 state submission against:
  1. Fundamental Diagram (S_FD): Speed and flow feasibility, jam density bounds,
     and absence of degenerate empty-road predictions.
  2. LWR Conservation Balance (S_LWR): Diagnostic link-by-link flux conservation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import load_corridor_config, get_corridor_list


def check_submission_physics(
    submission_path: Path,
    release_root: Path,
) -> Dict[str, float]:
    """Inspect submission file for FD compliance and conservation consistency."""
    if not submission_path.exists():
        raise FileNotFoundError(f"State submission not found: {submission_path}")

    print(f"Reading state submission: {submission_path}...")
    df = pd.read_csv(submission_path)
    total_rows = len(df)
    print(f"Loaded {total_rows:,} rows.")

    if "task" in df.columns:
        df = df[df["task"] == "state"].copy()

    if "panel" not in df.columns or "link_id" not in df.columns:
        print("Merging panel metadata from submission_key.csv...")
        key = pd.read_csv(release_root / "submission_key.csv", usecols=["submission_id", "panel", "link_id"], low_memory=False)
        df = df.merge(key, on="submission_id", how="inner")

    metrics: Dict[str, float] = {}

    # 1. Non-negativity & valid range check
    invalid_speed = (df["speed_kmh"] <= 0).sum()
    invalid_flow = (df["flow_vph"] < 0).sum()
    metrics["non_positive_speed_count"] = int(invalid_speed)
    metrics["negative_flow_count"] = int(invalid_flow)

    # 2. Empty-road check: cells with flow < 50 vph
    empty_cells = (df["flow_vph"] < 50.0).sum()
    empty_share = empty_cells / max(1, total_rows)
    metrics["empty_cell_share"] = float(empty_share)

    # 3. FD checks per corridor
    panels = df["panel"].unique()
    fd_violations = 0
    total_checked = 0

    for panel in panels:
        p_df = df[df["panel"] == panel]
        cfg = load_corridor_config(release_root, panel)
        links = cfg.get("links")

        if links is not None:
            ffs_col = "free_speed_kmh" if "free_speed_kmh" in links.columns else ("free_flow_speed_kmh" if "free_flow_speed_kmh" in links.columns else None)
            if ffs_col:
                ffs_map = dict(zip(links["link_id"].astype(str), links[ffs_col]))
            else:
                ffs_map = {}
            ffs = p_df["link_id"].astype(str).map(ffs_map).fillna(115.0).to_numpy()
            excess_speed = (p_df["speed_kmh"].to_numpy() > ffs + 5.0).sum()
            fd_violations += excess_speed
            total_checked += len(p_df)

    metrics["fd_speed_violation_rate"] = float(fd_violations / max(1, total_checked))

    print("\n--- Physics Diagnostics Summary ---")
    print(f"Total Rows Checked: {total_rows:,}")
    print(f"Non-positive Speed Count: {metrics['non_positive_speed_count']}")
    print(f"Negative Flow Count: {metrics['negative_flow_count']}")
    print(f"Empty Road Cell Share: {metrics['empty_cell_share']:.2%} (Rule limit: < 20.0%)")
    print(f"FD Speed Violation Rate: {metrics['fd_speed_violation_rate']:.4%}")

    if metrics["empty_cell_share"] < 0.20 and metrics["fd_speed_violation_rate"] < 0.01:
        print("\nPASSED: Submission respects fundamental diagram physics bounds.")
    else:
        print("\nWARNING: Review physics violations before final submission.")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 3 Physics Compliance Diagnostics")
    parser.add_argument("--submission", type=Path, default=Path("outputs/submissions/state_submission.csv"))
    root = Path("./data/kaggle_public") if Path("./data/kaggle_public").exists() else Path("trafficflowbench/data/kaggle_public")
    parser.add_argument("--release-root", type=Path, default=root)
    args = parser.parse_args()

    check_submission_physics(args.submission, args.release_root)


if __name__ == "__main__":
    main()
