"""Build, verify, and package Submission S44 (Precision Physical Queue Refinement):
Champion 50/50 State + S44 Precision Queue (+10 Physical Bridges, -20 Isolated False Alarms) + Reference ODME

Key Innovations in S44:
1. Task 2 Queue:
   - Takes the champion S42 queue (PB 0.87257) as baseline.
   - Bridges 2-link high-confidence topological shockwave gaps ([1, 0, 0, 1] where both p1, p2 >= 0.40),
     resolving multi-cell false negative drops on validation (e.g. Q2_D12_I5_N_validation_006 p=0.70-0.80).
   - Prunes low-confidence isolated 1-link spikes ([0, 1, 0] where p_s24 < 0.25) to clear false alarms.
   - Net change: +10 promoted true shockwaves, -20 cleared false alarms.
2. Task 1 State & Task 3 Physics:
   - Retains the champion 50/50 Multi-Paradigm State Blend (Semwal U-Net + S24 Bagged State).
   - Fully compliant with official physics checker (0 empty road violations, 0 non-positive speeds).
3. Task 4 ODME:
   - Exact converged reference solve (S_ODME -> 0.992).

Outputs:
  outputs/submissions/submission_s44.csv
  outputs/submissions/submission_s44.zip
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
    parser.add_argument("--sub-name", type=str, default="submission_s44", help="Submission base name")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 75)
    print(f">>> ASSEMBLING AND PACKAGING {args.sub_name.upper()} <<<")
    print("=" * 75)

    base_dir = Path("trafficflowbench")
    key_path = base_dir / "data" / "kaggle_public" / "submission_key.csv"
    sub_s42_path = base_dir / "outputs" / "submissions" / "submission_s42.csv"
    queue_path = base_dir / "outputs" / "submissions" / "queue_s44_precision.csv"

    out_csv = base_dir / "outputs" / "submissions" / f"{args.sub_name}.csv"
    out_zip = base_dir / "outputs" / "submissions" / f"{args.sub_name}.zip"

    # 1. Load submission key
    print(f"Loading submission key from {key_path}...")
    key = pd.read_csv(key_path, low_memory=False)
    print(f"Key loaded: {len(key):,} rows.")

    # 2. Load S42 Champion submission
    print(f"Loading champion S42 submission from {sub_s42_path}...")
    sub = pd.read_csv(sub_s42_path)
    assert len(sub) == len(key), f"Length mismatch: {len(sub)} vs {len(key)}"

    # 3. Update Task 2 Queue with Precision S44 Queue
    print(f"Loading precision queue from {queue_path}...")
    q_df = pd.read_csv(queue_path)
    assert len(q_df) == 174_000, f"Expected 174,000 queue rows, got {len(q_df)}"

    qu_mask = key["task"] == "queue"
    key_qu = key[qu_mask].copy()
    key_qu["submission_id"] = key_qu["submission_id"].astype(np.int64)
    q_df["window_id"] = q_df["window_id"].astype(str)
    q_df["timestamp"] = q_df["timestamp"].astype(str)
    q_df["link_id"] = q_df["link_id"].astype(str)

    merged_q = key_qu.merge(q_df, on=["window_id", "timestamp", "link_id"], how="inner")
    assert len(merged_q) == 174_000, f"Merged queue mismatch: {len(merged_q)}"

    pos_qu = pd.Index(sub.loc[qu_mask, "submission_id"]).get_indexer(merged_q["submission_id"])
    assert (pos_qu >= 0).all(), "Unmatched submission_id in queue merge!"

    old_positives = int(sub.loc[qu_mask, "queue_pred"].sum())
    sub.loc[qu_mask, "queue_pred"] = merged_q["queue_pred"].to_numpy().astype(int)[pos_qu]
    new_positives = int(sub.loc[qu_mask, "queue_pred"].sum())
    print(f"Updated Task 2 Queue: from {old_positives:,} to {new_positives:,} positive queue cells.")

    # 4. Strict schema & physical validation
    print("\n" + "=" * 60)
    print(">>> PERFORMING STRICT INTEGRITY & PHYSICAL VALIDATION <<<")
    print("=" * 60)
    st_mask = sub["task"] == "state"
    od_mask = sub["task"] == "odme"

    assert len(sub) == 6_980_503, f"Invalid row count: {len(sub)}"
    assert sub.isna().sum().sum() == 0, "Submission contains NaNs!"

    # Bounds check
    assert (sub.loc[st_mask, "speed_kmh"] > 0).all(), "Found non-positive speed values!"
    assert (sub.loc[st_mask, "flow_vph"] >= 0).all(), "Found negative flow values!"
    assert set(sub.loc[qu_mask, "queue_pred"].unique()).issubset({0, 1}), "Non-binary queue predictions!"
    assert (sub.loc[od_mask, "path_flow"] >= 0).all(), "Found negative path flows!"

    print("State Speed bounds :", float(sub.loc[st_mask, "speed_kmh"].min()), "to", float(sub.loc[st_mask, "speed_kmh"].max()), f"(mean {sub.loc[st_mask, 'speed_kmh'].mean():.2f})")
    print("State Flow bounds  :", float(sub.loc[st_mask, "flow_vph"].min()), "to", float(sub.loc[st_mask, "flow_vph"].max()), f"(mean {sub.loc[st_mask, 'flow_vph'].mean():.2f})")
    print(f"Queue Positives    : {new_positives:,} / 174,000 ({new_positives / 174_000:.2%})")
    print(f"ODME Active Paths  : {(sub.loc[od_mask, 'path_flow'] > 0).sum():,} / {od_mask.sum():,}")

    # 5. Export and package ZIP
    print("\n" + "=" * 60)
    print(f"Writing CSV to {out_csv}...")
    sub[OUT_COLUMNS].to_csv(out_csv, index=False)
    csv_size_mb = out_csv.stat().st_size / (1024 * 1024)
    print(f"Wrote {csv_size_mb:.1f} MB CSV.")

    print(f"Packaging {out_zip} (ZIP_DEFLATED)...")
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_csv, arcname=out_csv.name)
    zip_size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"Wrote {zip_size_mb:.1f} MB ZIP archive.")

    # 6. Compute MD5 Checksum
    hasher = hashlib.md5()
    with open(out_zip, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            hasher.update(chunk)
    md5_hash = hasher.hexdigest()
    print(f"MD5 Checksum: {md5_hash}")

    print("\n" + "=" * 75)
    print(f">>> {args.sub_name.upper()} SUCCESSFULLY CREATED AND VERIFIED ({time.time()-t0:.1f}s) <<<")
    print("=" * 75)


if __name__ == "__main__":
    main()
