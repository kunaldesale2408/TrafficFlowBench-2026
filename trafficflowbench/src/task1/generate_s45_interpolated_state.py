"""Generate Task 1 Continuous Physical State (S45):
Convex Blend of Champion Multi-Paradigm State (S44) + Exact Linear Temporal Inpainting on Bracketed Cells.

Architecture:
1. Base Anchor:
   - S44 Champion State (PB 0.87287: 50% Semwal U-Net/Trees + 50% S24 Bagged Deep-LGB/CatBoost).
   - Provides global spatial shockwave context and high-capacity tabular features on long/unbracketed gaps.
2. Continuous Physical Inpainting:
   - Evaluates exact 5-minute linear interpolation along each detector link across all 610 daily corridor matrices.
   - Operates strictly on bracketed interior gaps (limit_area='inside'), covering ~98.7% of all scored target cells.
3. Convex Simplex Integration (w_temp = 0.60, w_anchor = 0.40):
   - Smooths step-function discretization variance while preserving spatial wave topology.
   - Unbracketed day-edge cells (1.3%) fall back 100% to champion deep multi-paradigm anchor.
4. Strict Physical Guardrails:
   - Hard jam density lower bound: v >= q / k_jam.
   - Velocity feasibility: v in [5.0, 1.05 * v_f] km/h.
   - Flow feasibility: q in [50.0, 1.02 * capacity] vph.
   - Zero empty-road (< 50 vph) or non-positive speed artifacts.

Outputs:
  outputs/submissions/state_s45_interpolated_fd.csv (6,735,795 rows)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("D:/Kaggle/TrafficFlowBench 2026/trafficflowbench")
DATA = ROOT / "data" / "kaggle_public"
SUB_S44_PATH = ROOT / "outputs" / "submissions" / "submission_s44.csv"
OUT_CSV = ROOT / "outputs" / "submissions" / "state_s45_interpolated_fd.csv"


def load_network_fd_params() -> tuple[dict, dict, dict]:
    ffs_map, cap_map, jam_map = {}, {}, {}
    corridors_dir = DATA / "corridors"
    for p in corridors_dir.iterdir():
        if p.is_dir():
            fd_file = p / "network" / "fd_parameters.csv"
            if fd_file.exists():
                fdf = pd.read_csv(fd_file, dtype={"link_id": str})
                for _, r in fdf.iterrows():
                    key = (p.name, str(r["link_id"]))
                    lanes = float(r.get("lanes", 3.0))
                    ffs_map[key] = float(r.get("free_speed_kmh", 105.0))
                    cap_map[key] = float(r.get("capacity_vph", 8000.0))
                    jam_map[key] = float(r.get("k_jam", 120.0 * lanes))
    return ffs_map, cap_map, jam_map


def main() -> None:
    t0 = time.time()
    print("=" * 75)
    print(">>> GENERATING TASK 1 CONTINUOUS PHYSICAL STATE (S45) <<<")
    print("=" * 75)

    # 1. Load submission key
    key_path = DATA / "submission_key.csv"
    print(f"Loading submission key from {key_path}...")
    key = pd.read_csv(key_path, low_memory=False)
    st_mask = key["task"] == "state"
    key_st = key[st_mask].copy()
    print(f"Key state rows: {len(key_st):,}")

    # 2. Load S44 champion state anchor
    print(f"Loading champion S44 submission from {SUB_S44_PATH}...")
    sub_s44 = pd.read_csv(SUB_S44_PATH)
    assert len(sub_s44) == len(key), f"Length mismatch: {len(sub_s44)} vs {len(key)}"

    v_anchor = sub_s44.loc[st_mask, "speed_kmh"].to_numpy().astype(np.float32)
    q_anchor = sub_s44.loc[st_mask, "flow_vph"].to_numpy().astype(np.float32)
    key_st["v_anchor"] = v_anchor
    key_st["q_anchor"] = q_anchor

    # 3. Load network physical FD parameters
    print("Loading network FD parameters...")
    ffs_map, cap_map, jam_map = load_network_fd_params()

    # 4. Compute exact linear temporal inpainting across all corridor files
    print("\nComputing continuous linear temporal inpainting across all 10 corridors...")
    corridors = sorted([p.name for p in (DATA / "corridors").iterdir() if p.is_dir()])
    splits = ("validation", "private")

    interp_pieces = []
    total_files = 0

    for panel in corridors:
        t_panel = time.time()
        p_files_count = 0
        for split in splits:
            split_dir = DATA / "corridors" / panel / split / "mainline_states_masked"
            files = sorted(list(split_dir.glob("**/*.parquet")))
            p_files_count += len(files)

            for f in files:
                df = pd.read_parquet(f, columns=["timestamp", "link_id", "speed_kmh", "flow_vph"])
                times = sorted(df["timestamp"].unique())
                links = df["link_id"].unique()

                v_mat = df.pivot(index="timestamp", columns="link_id", values="speed_kmh").reindex(index=times, columns=links)
                q_mat = df.pivot(index="timestamp", columns="link_id", values="flow_vph").reindex(index=times, columns=links)

                v_interp = v_mat.interpolate(axis=0, limit_area="inside").stack().rename("temp_v").reset_index()
                q_interp = q_mat.interpolate(axis=0, limit_area="inside").stack().rename("temp_q").reset_index()

                m = v_interp.merge(q_interp, on=["timestamp", "link_id"])
                m["panel"] = panel
                m["link_id"] = m["link_id"].astype(str)
                interp_pieces.append(m)

        total_files += p_files_count
        print(f"  {panel:12s}: processed {p_files_count} daily files in {time.time() - t_panel:.1f}s")

    print(f"\nConcatenating inpainting matrices ({total_files} total files)...")
    all_interp = pd.concat(interp_pieces, ignore_index=True)
    all_interp["link_id"] = all_interp["link_id"].astype(str)
    all_interp["panel"] = all_interp["panel"].astype(str)
    all_interp["timestamp"] = all_interp["timestamp"].astype(str)
    print(f"Total interpolated cells: {len(all_interp):,}")

    # 5. Merge inpainting onto state target keys
    print("Merging continuous inpainting with state target keys...")
    key_st["link_id"] = key_st["link_id"].astype(str)
    key_st["panel"] = key_st["panel"].astype(str)
    key_st["timestamp"] = key_st["timestamp"].astype(str)

    merge_keys = ["panel", "timestamp", "link_id"]
    merged = key_st.merge(all_interp[merge_keys + ["temp_v", "temp_q"]], on=merge_keys, how="left")
    assert len(merged) == len(key_st), f"Merge row count mismatch: {len(merged)} vs {len(key_st)}"

    temp_v = merged["temp_v"].to_numpy().astype(np.float32)
    temp_q = merged["temp_q"].to_numpy().astype(np.float32)
    v_anc = merged["v_anchor"].to_numpy().astype(np.float32)
    q_anc = merged["q_anchor"].to_numpy().astype(np.float32)

    is_bracketed = np.isfinite(temp_v) & np.isfinite(temp_q)
    n_bracketed = int(is_bracketed.sum())
    print(f"Bracketed target cells: {n_bracketed:,} / {len(merged):,} ({n_bracketed / len(merged):.2%})")

    # 6. Apply Optimal Simplex Combination (w_temp = 0.60, w_anchor = 0.40 on bracketed)
    w_temp = 0.60
    w_anc = 1.0 - w_temp
    print(f"\nApplying convex blend: {w_temp:.0%} Inpainting + {w_anc:.0%} S44 Anchor on bracketed cells...")

    v_blend = np.where(is_bracketed, w_temp * temp_v + w_anc * v_anc, v_anc)
    q_blend = np.where(is_bracketed, w_temp * temp_q + w_anc * q_anc, q_anc)

    # 7. Apply Strict Physical Feasibility & Jam Density Constraints
    print("Enforcing physical Fundamental Diagram feasibility envelopes...")
    pairs = list(zip(merged["panel"], merged["link_id"]))
    ffs_arr = np.array([ffs_map.get(k, 105.0) for k in pairs], dtype=np.float32)
    cap_arr = np.array([cap_map.get(k, 8000.0) for k in pairs], dtype=np.float32)
    jam_arr = np.array([jam_map.get(k, 450.0) for k in pairs], dtype=np.float32)

    # Hard jam density lower bound: v >= q / k_jam
    min_speed_jam = q_blend / (jam_arr + 1e-6)
    v_final = np.clip(v_blend, np.maximum(5.0, min_speed_jam), ffs_arr + 5.0)
    q_final = np.clip(q_blend, 50.0, cap_arr * 1.02)

    # Verify no NaNs or invalid values
    assert np.isfinite(v_final).all(), "Found NaNs in speed!"
    assert np.isfinite(q_final).all(), "Found NaNs in flow!"
    assert (v_final > 0).all(), "Found non-positive speed!"
    assert (q_final >= 50.0).all(), "Found flow < 50 vph!"

    diff_v = v_final - v_anc
    diff_q = q_final - q_anc
    print(f"\nDelta vs S44 Champion State:")
    print(f"  Speed: RMSE = {np.sqrt(np.mean(diff_v**2)):.2f} km/h, Mean bias = {np.mean(diff_v):.2f} km/h, Max diff = {np.max(np.abs(diff_v)):.2f} km/h")
    print(f"  Flow : RMSE = {np.sqrt(np.mean(diff_q**2)):.2f} vph,  Mean bias = {np.mean(diff_q):.2f} vph,  Max diff = {np.max(np.abs(diff_q)):.2f} vph")
    print(f"  Speed correlation with S44: {np.corrcoef(v_final, v_anc)[0, 1]:.5f}")
    print(f"  Flow  correlation with S44: {np.corrcoef(q_final, q_anc)[0, 1]:.5f}")

    # 8. Assemble output DataFrame
    out_cols = ["panel", "timestamp", "station_id", "link_id", "mask_regime", "speed_kmh", "flow_vph"]
    out_df = pd.DataFrame({
        "panel": merged["panel"].to_numpy(),
        "timestamp": merged["timestamp"].to_numpy(),
        "station_id": merged["station_id"].to_numpy(),
        "link_id": merged["link_id"].to_numpy(),
        "mask_regime": merged["mask_regime"].to_numpy(),
        "speed_kmh": v_final.astype(np.float32),
        "flow_vph": q_final.astype(np.float32),
    })

    print(f"\nSaving S45 state to {OUT_CSV}...")
    out_df.to_csv(OUT_CSV, index=False, float_format="%.4f")
    csv_size_mb = OUT_CSV.stat().st_size / (1024 * 1024)
    print(f"Successfully saved {len(out_df):,} state rows ({csv_size_mb:.1f} MB) in {time.time() - t0:.1f}s.")
    print("=" * 75)


if __name__ == "__main__":
    main()
