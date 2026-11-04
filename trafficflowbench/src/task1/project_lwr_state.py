"""Projected LWR Kinematic State Generator:
Applies the Log-Space Kinematic Wave Conservation Anchor and Decoupled Simplex
to combine Semwal U-Net (CV) and S24 Multi-Seed Bagged Engine (Trees):
  1. Decoupled Convex Simplex:
     - Speed: 55% S24 Champion Engine + 45% Semwal U-Net
     - Flow:  60% S24 Champion Engine + 40% Semwal U-Net
  2. Localized Same-Day Vehicle Accumulation Projection (interp_N_grid):
     - Anchors masked cells strictly to same-day observed accumulation N = (q/v)*L.
     - Damped log-density projection with run-length calibration: w=0.12 (run 1), 0.08 (run 2), 0.04 (run 3-4).
     - Split ratio alpha=0.85 (85% adjustment in flow, 15% in speed).
  3. Calibrated Triangular Fundamental Diagram & Hard Jam Density Enforcement:
     - Hard lower bound: v >= q / k_jam (100% compliant with official physics checker).
     - Congested envelope guardrail: q <= q_fd_max(v).
     - Flow floor >= 100.0 vph (zero empty road penalty).
     - Speed bounds: v in [5.0, v_f + 5.0] km/h.
Outputs:
  outputs/submissions/state_lwr_projected.csv (6,735,795 rows)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("trafficflowbench").resolve()
DATA = ROOT / "data" / "kaggle_public"
ASSETS = ROOT / "kaggle_research" / "anubis_assets"
SEM_VAL = ROOT / "kaggle_research" / "semwal_087" / "output" / "task1" / "task1_validation.csv"
SEM_PRIV = ROOT / "kaggle_research" / "semwal_087" / "output" / "task1" / "task1_private.csv"
S24_STATE = ROOT / "outputs" / "submissions" / "state_s24_ensemble_fd.csv"
KEY_FILE = DATA / "submission_key.csv"
OUT_DIR = ROOT / "outputs" / "submissions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["TFB_DATA_ROOT"] = str(DATA)
os.environ["TFB_OUTPUT_ROOT"] = str(ROOT / "scratch")
os.environ["TFB_THREADS"] = "6"
os.environ["TFB_PREDICT_DEVICE"] = "CPU"

sys.path.insert(0, str(ASSETS))
import state_model as base


def load_network_fd_params() -> tuple[dict, dict, dict, dict]:
    ffs_map, cap_map, jam_map, len_map = {}, {}, {}, {}
    corridors_dir = DATA / "corridors"
    for p in corridors_dir.iterdir():
        if p.is_dir():
            fd_file = p / "network" / "fd_parameters.csv"
            links_file = p / "network" / "links.csv"
            if links_file.exists():
                ldf = pd.read_csv(links_file, dtype={"link_id": str})
                for _, r in ldf.drop_duplicates("link_id").iterrows():
                    len_map[(p.name, str(r["link_id"]))] = float(r.get("length_km", 1.0))

            if fd_file.exists():
                fdf = pd.read_csv(fd_file, dtype={"link_id": str})
                for _, r in fdf.iterrows():
                    key = (p.name, str(r["link_id"]))
                    lanes = float(r.get("lanes", 3.0))
                    ffs_map[key] = float(r.get("free_speed_kmh", 105.0))
                    cap_map[key] = float(r.get("capacity_vph", 8000.0))
                    jam_map[key] = float(r.get("k_jam", 120.0 * lanes))
    return ffs_map, cap_map, jam_map, len_map


def interp_N_grid(obs_v: np.ndarray, obs_q: np.ndarray, elig: np.ndarray, L_vec: np.ndarray, rows: np.ndarray, cols: np.ndarray):
    """Linearly interpolate vehicle accumulation N = (q / v) * L strictly within the same day."""
    ok = elig & np.isfinite(obs_v) & np.isfinite(obs_q) & (obs_v > 1.0)
    N = np.where(ok, (obs_q / np.maximum(obs_v, 1.0)) * L_vec[None, :], np.nan)
    T = N.shape[0]
    pos = np.arange(T, dtype=np.int32)[:, None]
    prev = np.maximum.accumulate(np.where(ok, pos, -1), axis=0)
    nxt = np.minimum.accumulate(np.where(ok, pos, T)[::-1], axis=0)[::-1]
    p, n = prev[rows, cols], nxt[rows, cols]

    # Must be on the exact same day (288 slots per day)
    has = (p >= 0) & (n < T) & ((rows // 288) == (np.maximum(p, 0) // 288)) & ((rows // 288) == (np.minimum(n, T - 1) // 288))
    pp, nn = np.maximum(p, 0), np.minimum(n, T - 1)
    t_frac = (rows - pp) / np.maximum(nn - pp, 1)
    Ni = N[pp, cols] * (1.0 - t_frac) + N[nn, cols] * t_frac
    return np.where(has, Ni, np.nan), np.where(has, nn - pp - 1, 10**6)


def main():
    t0 = time.time()
    print("=" * 75)
    print(">>> GENERATING PROJECTED LWR KINEMATIC STATE (S39 STATE CORE) <<<")
    print("=" * 75)

    # 1. Load Network & FD Parameters
    print("Loading network parameters & FD envelopes across 10 corridors...")
    ffs_map, cap_map, jam_map, len_map = load_network_fd_params()
    print(f"Loaded network maps: {len(len_map):,} corridor links.")

    # 2. Load Semwal U-Net predictions
    print(f"Loading Semwal U-Net / Trees predictions ({SEM_VAL.name} + {SEM_PRIV.name})...")
    t1_val = pd.read_csv(SEM_VAL)
    t1_priv = pd.read_csv(SEM_PRIV)
    sem_raw = pd.concat([t1_val, t1_priv], ignore_index=True)
    print(f"Loaded Semwal raw state: {len(sem_raw):,} rows.")

    # 3. Load S24 Champion Bagged State
    print(f"Loading S24 Champion State ({S24_STATE.name})...")
    s24_raw = pd.read_csv(S24_STATE)
    print(f"Loaded S24 raw state: {len(s24_raw):,} rows.")

    # 4. Align with Submission Key
    print(f"Loading submission key from {KEY_FILE}...")
    key = pd.read_csv(KEY_FILE, low_memory=False)
    key_st = key[key["task"] == "state"].copy()
    print(f"Key state rows: {len(key_st):,}")

    keys = ["panel", "timestamp", "station_id", "link_id", "mask_regime"]
    for k in keys:
        sem_raw[k] = sem_raw[k].astype(str)
        s24_raw[k] = s24_raw[k].astype(str)
        key_st[k] = key_st[k].astype(str)

    print("Merging predictions against master key...")
    m_sem = key_st.merge(sem_raw[keys + ["speed_kmh", "flow_vph"]], on=keys, how="left")
    m_s24 = key_st.merge(s24_raw[keys + ["speed_kmh", "flow_vph"]], on=keys, how="left")

    assert m_sem["speed_kmh"].notna().all(), "Missing values in Semwal merge!"
    assert m_s24["speed_kmh"].notna().all(), "Missing values in S24 merge!"

    # 5. Pareto-Optimal Decoupled Simplex Combination
    # Speed: 55% S24 + 45% Semwal (balances U-Net spatial smoothness with multi-seed bagged trees)
    # Flow:  60% S24 + 40% Semwal (leverages CatBoost d10 conservation and FD compliance)
    wv, wq = 0.55, 0.60
    print(f"Computing Decoupled Blend: Speed={wv*100:.0f}% S24 / {(1-wv)*100:.0f}% Semwal | Flow={wq*100:.0f}% S24 / {(1-wq)*100:.0f}% Semwal...")

    v_blend = (1.0 - wv) * m_sem["speed_kmh"].to_numpy() + wv * m_s24["speed_kmh"].to_numpy()
    q_blend = (1.0 - wq) * m_sem["flow_vph"].to_numpy() + wq * m_s24["flow_vph"].to_numpy()

    # Base safety bounds
    v_blend = np.clip(v_blend, 5.0, 140.0)
    q_blend = np.clip(q_blend, 50.0, 150000.0)

    key_st["speed_kmh"] = v_blend
    key_st["flow_vph"] = q_blend

    # 6. Apply Localized Same-Day Accumulation & FD Projection per corridor
    print("\nApplying Log-Space Vehicle Accumulation Projection & FD Constraints per corridor...")
    panels = base.all_panels()
    splits = ("validation", "private")

    out_pieces = []
    total_anchored = 0
    total_processed = 0

    for split in splits:
        for pi, name in enumerate(panels):
            t_p = time.time()
            panel = base.Panel(name, pi)
            frame = base.month_frame(name, "", split=split)
            times, v_obs, q_obs, e_obs = base.matrices(frame, panel)

            # Extract targets for this panel and split
            sub_panel = key_st[(key_st["panel"] == name) & (key_st["timestamp"].str[:7] == ("2031-03" if split == "validation" else "2031-04"))].copy()
            if sub_panel.empty:
                continue

            rows = times.get_indexer(sub_panel["timestamp"])
            cols = pd.Index(panel.links).get_indexer(sub_panel["link_id"])
            assert (rows >= 0).all() and (cols >= 0).all()

            v_curr = sub_panel["speed_kmh"].to_numpy(dtype=np.float64)
            q_curr = sub_panel["flow_vph"].to_numpy(dtype=np.float64)

            # Network parameters for this panel's links
            links = sub_panel["link_id"].astype(str).to_numpy()
            pairs = [(name, lid) for lid in links]
            ffs_arr = np.array([ffs_map.get(k, 105.0) for k in pairs], dtype=np.float64)
            cap_arr = np.array([cap_map.get(k, 8000.0) for k in pairs], dtype=np.float64)
            jam_arr = np.array([jam_map.get(k, 450.0) for k in pairs], dtype=np.float64)
            len_arr = np.array([len_map.get(k, 1.0) for k in pairs], dtype=np.float64)
            L_panel = np.array([len_map.get((name, str(lid)), 1.0) for lid in panel.links], dtype=np.float64)

            # Step A: Interpolate accumulation N strictly within same day
            Ni, run_len = interp_N_grid(v_obs, q_obs, e_obs, L_panel, rows, cols)

            k_hat = q_curr / np.maximum(v_curr, 1.0)
            k_i = Ni / len_arr
            ok_n = np.isfinite(k_i) & (k_i > 0) & (k_hat > 0) & (run_len <= 4)

            # Calibrated dampening weights: w=0.12 (run 1), 0.08 (run 2), 0.04 (run 3-4)
            w_eff = np.where(run_len == 1, 0.12, np.where(run_len == 2, 0.08, np.where(run_len <= 4, 0.04, 0.0)))
            delta_log = np.where(ok_n, (np.log(np.maximum(k_i, 1e-9)) - np.log(np.maximum(k_hat, 1e-9))) * w_eff, 0.0)

            # Allocation: alpha=0.85 (85% adjustment to flow, 15% to speed to preserve speed RMSE)
            alpha_d = 0.85
            v_proj = np.clip(v_curr * np.exp(-(1.0 - alpha_d) * delta_log), 5.0, ffs_arr + 5.0)
            q_proj = np.clip(q_curr * np.exp(alpha_d * delta_log), 100.0, cap_arr * 1.02)

            # Step B: Triangular Fundamental Diagram congested upper envelope
            kc_arr = cap_arr / np.maximum(ffs_arr, 1.0)
            w_wave = cap_arr / np.maximum(jam_arr - kc_arr, 10.0)
            q_fd_max = np.minimum(1.04 * cap_arr, (w_wave * jam_arr) / (1.0 + w_wave / np.maximum(v_proj, 5.0)))
            congested_excess = (v_proj < 0.60 * ffs_arr) & (q_proj > q_fd_max)
            q_proj = np.where(congested_excess, 0.95 * q_proj + 0.05 * q_fd_max, q_proj)

            # Step C: Hard Jam Density Constraint (v >= q / k_jam)
            min_v_final = q_proj / (jam_arr + 1e-6)
            v_final = np.clip(v_proj, np.maximum(5.0, min_v_final), ffs_arr + 5.0)
            q_final = np.clip(q_proj, 100.0, cap_arr)

            sub_panel["speed_kmh"] = v_final
            sub_panel["flow_vph"] = q_final

            out_pieces.append(sub_panel)
            n_anch = int(ok_n.sum())
            total_anchored += n_anch
            total_processed += len(sub_panel)
            print(f"  {name} ({split}): {len(sub_panel):,} cells ({n_anch/len(sub_panel):.1%} anchored) in {time.time()-t_p:.1f}s")

    # 7. Concatenate & verify all corridors
    final_df = pd.concat(out_pieces, ignore_index=True)
    assert len(final_df) == len(key_st), f"Row count mismatch: {len(final_df):,} vs {len(key_st):,}"

    # Align perfectly back to key_st order
    print("\nAligning final rows to submission key order...")
    merged_final = key_st[["submission_id", "panel", "timestamp", "station_id", "link_id", "mask_regime"]].merge(
        final_df[["panel", "timestamp", "station_id", "link_id", "mask_regime", "speed_kmh", "flow_vph"]],
        on=["panel", "timestamp", "station_id", "link_id", "mask_regime"],
        how="inner",
        validate="one_to_one"
    )
    assert len(merged_final) == 6735795, f"Expected 6,735,795 rows, got {len(merged_final):,}"

    # Strict physical consistency checks
    v_out = merged_final["speed_kmh"].to_numpy()
    q_out = merged_final["flow_vph"].to_numpy()

    assert np.isfinite(v_out).all(), "Non-finite speed values!"
    assert np.isfinite(q_out).all(), "Non-finite flow values!"
    assert (v_out >= 5.0).all(), f"Found speed < 5 km/h: min={v_out.min():.2f}"
    assert (q_out >= 50.0).all(), f"Found flow < 50 vph: min={q_out.min():.2f}"
    assert float((q_out < 50.0).mean()) == 0.0, "Empty road floor violation!"

    out_csv = OUT_DIR / "state_lwr_projected.csv"
    print(f"\nWriting state predictions to {out_csv}...")
    cols_to_save = ["panel", "timestamp", "station_id", "link_id", "mask_regime", "speed_kmh", "flow_vph"]
    merged_final[cols_to_save].to_csv(out_csv, index=False, float_format="%.4f")
    csv_mb = out_csv.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 75)
    print(f"STATE GENERATION COMPLETE: {len(merged_final):,} rows in {time.time()-t0:.1f}s ({csv_mb:.1f} MB)")
    print(f"  Speed: min={v_out.min():.2f}, mean={v_out.mean():.2f}, max={v_out.max():.2f} km/h")
    print(f"  Flow:  min={q_out.min():.1f}, mean={q_out.mean():.1f}, max={q_out.max():.1f} vph")
    print(f"  Physical Anchoring Rate: {total_anchored / total_processed:.1%}")
    print("=" * 75)


if __name__ == "__main__":
    main()
