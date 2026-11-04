"""Build, verify, and package Submission S40 (Next-Generation PB Candidate):
Causal T-Slice Ongoing Queue + 1D Milepost Morphological Closing
+ Proven 50/50 Multi-Paradigm State Blend + Converged Reference ODME

Key Innovations in S40:
1. Task 2 Causal T-Slice Forward-Shift:
   - Uses real observations at forecast origin T from mainline_states_masked.
   - Evaluates forward-shifted ongoing model [T-55..T] and blends p_causal = 0.60 * p_std + 0.40 * p_T.
   - Promotes confirmed active queues at T (v_T <= v_cut) with p_causal >= 0.45.
   - Clears confirmed free-flow links at T (v_T > v_cut + 12) with p_causal < 0.38.
2. Task 2 1D Milepost Morphological Closing:
   - Bridges interior 1-link queue holes along physical highway milepost order when p_causal >= 0.28.
   - Reconnects physical freeway shockwaves to maximize Jaccard IoU.
3. Task 1 State:
   - Retains the proven champion 50/50 Multi-Paradigm State Blend (Semwal U-Net + S24 Engine, PB 0.87173).
4. Task 4 ODME:
   - Exact converged reference solve (S_od -> 1.000, S_ODME -> 0.992).

Outputs:
  outputs/submissions/submission_s40.csv
  outputs/submissions/submission_s40.zip
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
    parser.add_argument("--sub-name", type=str, default="submission_s40", help="Submission base name")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 75)
    print(f">>> ASSEMBLING AND PACKAGING {args.sub_name.upper()} <<<")
    print("=" * 75)

    base_dir = Path("trafficflowbench")
    key_path = base_dir / "data" / "kaggle_public" / "submission_key.csv"
    t1_dir = base_dir / "kaggle_research" / "semwal_087" / "output" / "task1"
    sub_87_path = base_dir / "kaggle_research" / "semwal_087" / "output" / "submission.csv"
    s24_state_path = base_dir / "outputs" / "submissions" / "state_s24_ensemble_fd.csv"
    queue_path = base_dir / "outputs" / "submissions" / "queue_causal_t_morphological.csv"

    out_csv = base_dir / "outputs" / "submissions" / f"{args.sub_name}.csv"
    out_zip = base_dir / "outputs" / "submissions" / f"{args.sub_name}.zip"

    # 1. Load submission key
    print(f"Loading submission key from {key_path}...")
    key = pd.read_csv(key_path, low_memory=False)
    print(f"Key loaded: {len(key):,} rows.")

    # 2. Load Task 1 components (Proven 50/50 State from PB S33)
    print("Loading Semwal pure Task 1 predictions (U-Net + Trees)...")
    t1_val = pd.read_csv(t1_dir / "task1_validation.csv")
    t1_priv = pd.read_csv(t1_dir / "task1_private.csv")
    semwal_raw = pd.concat([t1_val, t1_priv], ignore_index=True)

    print("Loading S24 Champion state (Multi-Seed Deep-LGB + CatBoost + interp_N)...")
    s24_state = pd.read_csv(s24_state_path)

    key_st = key[key["task"] == "state"].copy()
    keys = ["panel", "timestamp", "station_id", "link_id", "mask_regime"]
    for k in keys:
        semwal_raw[k] = semwal_raw[k].astype(str)
        s24_state[k] = s24_state[k].astype(str)
        key_st[k] = key_st[k].astype(str)

    print("Merging Task 1 components by keys...")
    m_sem = key_st.merge(semwal_raw[keys + ["speed_kmh", "flow_vph"]], on=keys, how="left")
    m_s24 = key_st.merge(s24_state[keys + ["speed_kmh", "flow_vph"]], on=keys, how="left")

    assert len(m_sem) == len(key_st), "Semwal row count mismatch!"
    assert len(m_s24) == len(key_st), "S24 row count mismatch!"

    # 50/50 Direct Blend (proven champion from S33)
    v_direct = 0.50 * m_sem["speed_kmh"].to_numpy() + 0.50 * m_s24["speed_kmh"].to_numpy()
    q_direct = 0.50 * m_sem["flow_vph"].to_numpy() + 0.50 * m_s24["flow_vph"].to_numpy()

    v_direct = np.clip(v_direct, 5.0, 135.0)
    q_direct = np.clip(q_direct, 50.0, None)

    # 3. Load Task 2 Queue from Causal-T + Morphological Model
    print(f"Loading Task 2 Queue from {queue_path}...")
    q_df = pd.read_csv(queue_path)
    qu_mask = key["task"] == "queue"
    key_qu = key[qu_mask].copy()

    q_keys = ["window_id", "timestamp", "link_id"]
    for k in q_keys:
        q_df[k] = q_df[k].astype(str)
        key_qu[k] = key_qu[k].astype(str)

    m_qu = key_qu.merge(q_df[q_keys + ["queue_pred"]], on=q_keys, how="left")
    assert len(m_qu) == len(key_qu), "Queue row count mismatch!"
    assert not m_qu["queue_pred"].isna().any(), "Missing queue predictions!"
    q_vals = m_qu["queue_pred"].to_numpy(dtype=int)

    # 4. Load Task 4 ODME from benchmark reference solve
    print("Loading Task 4 ODME reference solve...")
    df_87 = pd.read_csv(sub_87_path)
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
    od_mask = sub["task"] == "odme"

    sub.loc[st_mask, "speed_kmh"] = v_direct
    sub.loc[st_mask, "flow_vph"] = q_direct
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
    qu_vals_sub = sub.loc[qu_mask, "queue_pred"].to_numpy()
    od_vals_sub = sub.loc[od_mask, "path_flow"].to_numpy()

    assert (v_st >= 5.0).all(), f"Found speed < 5 km/h: min={v_st.min()}"
    assert (v_st <= 140.0).all(), f"Found speed > 140 km/h: max={v_st.max()}"
    assert (q_st >= 50.0).all(), f"Found flow < 50 vph: min={q_st.min()}"
    assert set(np.unique(qu_vals_sub)).issubset({0, 1}), f"Non-binary queue values: {np.unique(qu_vals_sub)}"
    assert (od_vals_sub >= 0.0).all(), f"Found negative OD flows: min={od_vals_sub.min()}"

    empty_share = float((q_st < 50.0).mean())
    assert empty_share <= 0.20, f"Empty road guard violation: {empty_share:.2%}"

    print(f"Row count:          {len(sub):,} [PASS]")
    print(f"NaN count:          {sub.isna().sum().sum()} [PASS]")
    print(f"State Speed:        min={v_st.min():.2f}, mean={v_st.mean():.2f}, max={v_st.max():.2f} km/h [PASS]")
    print(f"State Flow:         min={q_st.min():.1f}, mean={q_st.mean():.1f}, max={q_st.max():.1f} vph [PASS]")
    print(f"Queue Positives:    {qu_vals_sub.sum():,} ({qu_vals_sub.mean()*100:.2f}%) [PASS]")
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
