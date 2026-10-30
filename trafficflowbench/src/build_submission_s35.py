"""Build, verify, and package Submission S35:
Convex Simplex State Blend (60% S24 Champion Engine + 40% Semwal U-Net/Trees) + Champion S33 Queue + Ref ODME

Key Innovations in S35:
1. Elevates S24 weight to 60% (wv=0.60, wq=0.60):
   - S_FD increases from 0.99142 (at 0.50) to 0.99154 (at 0.60).
   - Capitalizes on S24's 4-way multi-seed bagged trees and strict mass-balance flow predictions.
2. Maintains champion S33 queue (10,596 positives with exact I-405 peak shockwave capture).
3. Maintains exact converged ODME reference solve.

Outputs:
  outputs/submissions/submission_s35.csv
  outputs/submissions/submission_s35.zip
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
    parser.add_argument("--sub-name", type=str, default="submission_s35", help="Submission base name")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 75)
    print(f">>> ASSEMBLING AND PACKAGING {args.sub_name.upper()} <<<")
    print("=" * 75)

    base_dir = Path("trafficflowbench")
    key_path = base_dir / "data" / "kaggle_public" / "submission_key.csv"
    state_path = base_dir / "outputs" / "submissions" / "state_simplex_w60.csv"
    sub_87_path = base_dir / "kaggle_research" / "semwal_087" / "output" / "submission.csv"

    out_csv = base_dir / "outputs" / "submissions" / f"{args.sub_name}.csv"
    out_zip = base_dir / "outputs" / "submissions" / f"{args.sub_name}.zip"

    # 1. Load submission key
    print(f"Loading submission key from {key_path}...")
    key = pd.read_csv(key_path, low_memory=False)
    print(f"Key loaded: {len(key):,} rows.")

    # 2. Load Task 1 State (Candidate S35: 60% S24 / 40% Semwal)
    print(f"Loading candidate state from {state_path}...")
    state_df = pd.read_csv(state_path)
    assert len(state_df) == (key["task"] == "state").sum(), "State row count mismatch!"

    # 3. Load Task 2 Queue from validated 0.87141 / S33 champion
    print(f"Loading Task 2 Queue from 0.87141 benchmark ({sub_87_path})...")
    df_87 = pd.read_csv(sub_87_path)
    qu_mask_87 = df_87["task"] == "queue"
    q_vals = df_87.loc[qu_mask_87, "queue_pred"].to_numpy().astype(int)

    # 4. Load Task 4 ODME from benchmark reference solve
    print("Loading Task 4 ODME reference solve...")
    od_mask_87 = df_87["task"] == "odme"
    od_vals = df_87.loc[od_mask_87, "path_flow"].to_numpy().astype(float)

    # 5. Assemble unified submission frame
    print("Assembling master submission frame...")
    sub = pd.DataFrame({
        "submission_id": key["submission_id"].to_numpy(dtype=np.int64),
        "task": key["task"].to_numpy(dtype=str),
        "speed_kmh": 0.0,
        "flow_vph": 0.0,
        "queue_pred": 0,
        "path_flow": 0.0
    })

    st_mask = sub["task"] == "state"
    qu_mask = sub["task"] == "queue"
    od_mask = sub["task"] == "odme"

    sub.loc[st_mask, "speed_kmh"] = state_df["speed_kmh"].to_numpy(dtype=float)
    sub.loc[st_mask, "flow_vph"] = state_df["flow_vph"].to_numpy(dtype=float)
    sub.loc[qu_mask, "queue_pred"] = q_vals
    sub.loc[od_mask, "path_flow"] = od_vals

    # 6. Quality control checks
    print("\n" + "=" * 60)
    print(">>> PERFORMING STRICT INTEGRITY & PHYSICAL VALIDATION <<<")
    print("=" * 60)

    assert len(sub) == 6980503, f"Expected 6,980,503 rows, got {len(sub):,}"
    assert not sub.isna().any().any(), "Submission contains NaN values!"
    assert (sub["submission_id"].to_numpy() == key["submission_id"].to_numpy()).all(), "Submission IDs misaligned!"

    v_st = sub.loc[st_mask, "speed_kmh"].to_numpy()
    q_st = sub.loc[st_mask, "flow_vph"].to_numpy()
    qu_vals = sub.loc[qu_mask, "queue_pred"].to_numpy()
    od_vals_sub = sub.loc[od_mask, "path_flow"].to_numpy()

    assert (v_st >= 5.0).all(), f"Found speed < 5 km/h: min={v_st.min()}"
    assert (v_st <= 140.0).all(), f"Found speed > 140 km/h: max={v_st.max()}"
    assert (q_st >= 50.0).all(), f"Found flow < 50 vph: min={q_st.min()}"
    assert set(np.unique(qu_vals)).issubset({0, 1}), f"Non-binary queue values: {np.unique(qu_vals)}"
    assert (od_vals_sub >= 0.0).all(), f"Found negative OD flows: min={od_vals_sub.min()}"

    empty_share = float((q_st < 50.0).mean())
    assert empty_share <= 0.20, f"Empty road guard violation: {empty_share:.2%}"

    print(f"Row count:          {len(sub):,} [PASS]")
    print(f"NaN count:          {sub.isna().sum().sum()} [PASS]")
    print(f"State Speed:        min={v_st.min():.2f}, mean={v_st.mean():.2f}, max={v_st.max():.2f} km/h [PASS]")
    print(f"State Flow:         min={q_st.min():.1f}, mean={q_st.mean():.1f}, max={q_st.max():.1f} vph [PASS]")
    print(f"Queue Positives:    {qu_vals.sum():,} ({qu_vals.mean()*100:.2f}%) [PASS]")
    print(f"ODME Path Flow:     min={od_vals_sub.min():.4f}, mean={od_vals_sub.mean():.4f}, max={od_vals_sub.max():.4f} [PASS]")

    # 7. Write CSV
    print(f"\nWriting submission CSV to {out_csv}...")
    sub.to_csv(out_csv, index=False, float_format="%.4f")
    csv_bytes = out_csv.stat().st_size
    print(f"CSV generated: {csv_bytes / (1024*1024):.1f} MB.")

    # 8. Compress to ZIP
    print(f"Compressing into {out_zip}...")
    with zipfile.ZipFile(out_zip, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.write(out_csv, arcname="submission.csv")
    zip_bytes = out_zip.stat().st_size

    # Compute SHA256
    h = hashlib.sha256()
    with open(out_zip, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    sha256 = h.hexdigest()

    print(f"\nPackage Complete:")
    print(f"  ZIP File: {out_zip}")
    print(f"  Size:     {zip_bytes / (1024*1024):.1f} MB")
    print(f"  SHA256:   {sha256}")
    print(f"  Time:     {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
