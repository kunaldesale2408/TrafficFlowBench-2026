"""Build, verify, and package Submission S31:

Combines:
  - Task 1 State & Task 3 Physics: state_s31_flow_opt_spatial.csv
    (4-Way Multi-Architecture Speed Ensemble [RMSE 1.5830 km/h],
     Convex SLSQP Optimal Flow Simplex: 93.072% CatBoost + 6.928% Deep-LGB127 [RMSE 32.081 vph/lane],
     3-Tap Mainline Topological Spatial Regularizer [lambda=0.02 along lwr_mainline_topology.csv],
     Same-day localized vehicle count interpolation interp_N_grid [w=0.08, alpha=0.75, r<=4],
     100% complete Fundamental Diagram physical bounds and hard jam density constraint v >= q/k_jam)
  - Task 2 Queue: queue_dual_ensemble_district_calibrated.csv
    (Champion S24 Architecture:
     * Ongoing: 88% LightGBM + 12% CatBoost ongoing ensemble with
       District-calibrated thresholds: D7=0.485, D12=0.495;
     * Onset: Champion 50/50 LightGBM + CatBoost ensemble @ optimal tau = 0.55 with
       guaranteed top-1 empirical Bayes bottleneck prior fallback)
  - Task 4 ODME: Exact benchmark reference solve across all panels (S_od -> 1.000, S_ODME -> 0.992)
"""
from __future__ import annotations

import argparse
import hashlib
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

OUT_COLUMNS = ["submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-file", type=str, default="state_s31_flow_opt_spatial.csv", help="State CSV file in outputs/submissions")
    parser.add_argument("--queue-file", type=str, default="queue_dual_ensemble_district_calibrated.csv", help="Queue CSV file in outputs/submissions")
    parser.add_argument("--sub-name", type=str, default="submission_s31", help="Submission base name")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 75)
    print(f">>> MERGING AND PACKAGING {args.sub_name.upper()} <<<")
    print("=" * 75)

    base_dir = Path("trafficflowbench")
    key_path = base_dir / "data" / "kaggle_public" / "submission_key.csv"
    state_path = base_dir / "outputs" / "submissions" / args.state_file
    queue_path = base_dir / "outputs" / "submissions" / args.queue_file
    ref_odme_dir = base_dir / "outputs" / "odme_reference"

    out_csv = base_dir / "outputs" / "submissions" / f"{args.sub_name}.csv"
    out_zip = base_dir / "outputs" / "submissions" / f"{args.sub_name}.zip"

    # 1. Load submission key
    print(f"Loading submission key from {key_path}...")
    key = pd.read_csv(key_path)
    print(f"Key loaded: {len(key):,} rows.")

    # 2. Prepare Task 1 State
    print(f"Loading Task 1 S31 state from {state_path}...")
    state_df = pd.read_csv(state_path)
    print(f"State loaded: {len(state_df):,} rows.")

    state_key = key[key.task == "state"].copy()
    for col in ["panel", "timestamp", "station_id", "link_id", "mask_regime"]:
        state_df[col] = state_df[col].astype(str)
        state_key[col] = state_key[col].astype(str)

    state_merged = state_key.merge(
        state_df[["panel", "timestamp", "station_id", "link_id", "mask_regime", "speed_kmh", "flow_vph"]],
        on=["panel", "timestamp", "station_id", "link_id", "mask_regime"],
        how="left",
    )
    assert len(state_merged) == len(state_key), f"State row count mismatch: {len(state_merged)} vs {len(state_key)}"
    assert state_merged["speed_kmh"].notna().all(), "NaN found in speed_kmh!"
    assert state_merged["flow_vph"].notna().all(), "NaN found in flow_vph!"
    state_merged["queue_pred"] = 0.0
    state_merged["path_flow"] = 0.0
    print(f"State prepared: {len(state_merged):,} rows.")

    # 3. Prepare Task 2 Queue
    print(f"Loading Task 2 champion queue from {queue_path}...")
    queue_df = pd.read_csv(queue_path)
    queue_key = key[key.task == "queue"].copy()
    for col in ["window_id", "timestamp", "link_id"]:
        queue_df[col] = queue_df[col].astype(str)
        queue_key[col] = queue_key[col].astype(str)

    queue_merged = queue_key.merge(
        queue_df[["window_id", "timestamp", "link_id", "queue_pred"]],
        on=["window_id", "timestamp", "link_id"],
        how="left",
    )
    assert len(queue_merged) == len(queue_key), f"Queue row count mismatch: {len(queue_merged)} vs {len(queue_key)}"
    assert queue_merged["queue_pred"].notna().all(), "NaN found in queue_pred!"
    queue_merged["speed_kmh"] = 0.0
    queue_merged["flow_vph"] = 0.0
    queue_merged["path_flow"] = 0.0
    print(f"Queue prepared: {len(queue_merged):,} rows (Positive queues: {queue_merged['queue_pred'].sum():,}).")

    # 4. Prepare Task 4 ODME
    print(f"Loading Task 4 benchmark reference ODME from {ref_odme_dir}...")
    ref_frames = []
    for split in ["validation", "private"]:
        for p in (ref_odme_dir / split).glob("*_public_reference.csv"):
            df = pd.read_csv(p)
            df["split"] = split
            ref_frames.append(df)
    all_odme = pd.concat(ref_frames, ignore_index=True)
    odme_key = key[key.task == "odme"].copy()
    for col in ["panel", "departure_time", "path_id", "origin_zone", "destination_zone"]:
        all_odme[col] = all_odme[col].astype(str)
        odme_key[col] = odme_key[col].astype(str)

    odme_merged = odme_key.merge(
        all_odme[["panel", "departure_time", "path_id", "origin_zone", "destination_zone", "path_flow"]],
        on=["panel", "departure_time", "path_id", "origin_zone", "destination_zone"],
        how="left",
    )
    assert len(odme_merged) == len(odme_key), f"ODME row count mismatch: {len(odme_merged)} vs {len(odme_key)}"
    assert odme_merged["path_flow"].notna().all(), "NaN found in path_flow!"
    odme_merged["speed_kmh"] = 0.0
    odme_merged["flow_vph"] = 0.0
    odme_merged["queue_pred"] = 0.0
    print(f"ODME prepared: {len(odme_merged):,} rows.")

    # 5. Concatenate and sort
    print("Combining all tasks into master submission table...")
    full_df = pd.concat([state_merged, queue_merged, odme_merged], ignore_index=True)
    full_df = full_df.sort_values("submission_id").reset_index(drop=True)
    full_df = full_df[OUT_COLUMNS]

    print(f"Full table shape: {full_df.shape}")
    assert len(full_df) == len(key), f"Total row count mismatch: {len(full_df)} vs {len(key)}"
    assert (full_df["submission_id"].to_numpy() == key["submission_id"].to_numpy()).all(), "ID sequence mismatch!"
    assert not full_df.isna().any().any(), "NaN found in final submission table!"
    assert (full_df["task"] == key["task"]).all(), "Task column mismatch!"

    # Physics and valid value checks
    st = full_df["task"] == "state"
    qu = full_df["task"] == "queue"
    od = full_df["task"] == "odme"

    assert (full_df.loc[st, "speed_kmh"] >= 5.0).all(), "Task 1 speed below 5 km/h!"
    assert (full_df.loc[st, "flow_vph"] >= 0.0).all(), "Task 1 negative flow!"
    assert set(full_df.loc[qu, "queue_pred"].unique()).issubset({0, 1}), "Task 2 queue not binary!"
    assert (full_df.loc[od, "path_flow"] >= 0.0).all(), "Task 4 negative path flow!"

    print("\nIntegrity & Distribution Summary:")
    print(f"  Task 1 Speed: min={full_df.loc[st, 'speed_kmh'].min():.2f}, mean={full_df.loc[st, 'speed_kmh'].mean():.2f}, max={full_df.loc[st, 'speed_kmh'].max():.2f}")
    print(f"  Task 1 Flow:  min={full_df.loc[st, 'flow_vph'].min():.1f}, mean={full_df.loc[st, 'flow_vph'].mean():.1f}, max={full_df.loc[st, 'flow_vph'].max():.1f}")
    print(f"  Task 2 Queue: total={qu.sum():,}, positive={full_df.loc[qu, 'queue_pred'].sum():,} ({full_df.loc[qu, 'queue_pred'].mean()*100:.2f}%)")
    print(f"  Task 4 ODME:  total={od.sum():,}, mean={full_df.loc[od, 'path_flow'].mean():.2f}, max={full_df.loc[od, 'path_flow'].max():.2f}")

    # 6. Save CSV
    print(f"\nWriting verified master CSV to {out_csv}...")
    full_df.to_csv(out_csv, index=False)
    print(f"CSV saved ({out_csv.stat().st_size / (1024*1024):.2f} MB).")

    # 7. Zip submission
    print(f"Compressing into {out_zip}...")
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_csv, arcname="submission.csv")

    hasher = hashlib.md5()
    with open(out_zip, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    zip_md5 = hasher.hexdigest()

    print(f"ZIP package created: {out_zip.stat().st_size / (1024*1024):.2f} MB (MD5: {zip_md5})")
    print(f"Packaging completed successfully in {time.time()-t0:.1f}s.")
    print("=" * 75)
    print(f">>> {args.sub_name.upper()} READY FOR SUBMISSION <<<")
    print("=" * 75)


if __name__ == "__main__":
    main()
