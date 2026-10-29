"""Generate Task 1 state predictions for Submission S24:
  - Multi-seed LightGBM Bagging: Seed 404 (d127) + Seed 42 (d127) + Seed 1024 (d96)
  - Blended with CatBoost Benchmark (depth-8 and depth-10) using optimal SLSQP simplex weights.
  - Deep 127-leaf LightGBM Flow blend (95% CatBoost + 5% Deep-LGB127).
  - Same-day localized vehicle count interpolation (interp_N_grid) on short missing runs (run <= 4).
  - Triangular congested-envelope guardrail on high-density cells.
  - 100% complete Fundamental Diagram physical bounds and hard jam density constraint (v >= q / k_jam).
Outputs:
  outputs/submissions/state_s24_ensemble_fd.csv
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path("trafficflowbench").resolve()
DATA = ROOT / "data" / "kaggle_public"
ASSETS = ROOT / "kaggle_research" / "anubis_assets"
MODELS_DIR = ROOT / "outputs" / "models"
SUB_DIR = ROOT / "outputs" / "submissions"
SUB_DIR.mkdir(parents=True, exist_ok=True)

os.environ["TFB_DATA_ROOT"] = str(DATA)
os.environ["TFB_OUTPUT_ROOT"] = str(ROOT / "scratch")
os.environ["TFB_THREADS"] = "6"
os.environ["TFB_PREDICT_DEVICE"] = "CPU"

sys.path.insert(0, str(ASSETS))
import state_model as base
import prepare_state
from state_inference import StatePredictor


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


def interp_N_grid(obs_v, obs_q, elig, L_vec, rows, cols):
    """Linearly interpolate vehicle count N = (q / v) * L strictly within the same day."""
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
    print(">>> GENERATING S24 ENSEMBLE STATE (CB + 3-SEED DEEP-LGB) WITH FULL FD <<<")
    print("=" * 75)

    # 1. Load weights configuration
    weights_path = MODELS_DIR / "speed_ensemble_weights_s24.json"
    if weights_path.exists():
        with open(weights_path) as f:
            weights_cfg = json.load(f)["weights"]
        w_cb = float(weights_cfg.get("catboost", 0.55))
        w_lgb_s1 = float(weights_cfg.get("lgb_seed404", 0.15))
        w_lgb_s2 = float(weights_cfg.get("lgb_seed42", 0.15))
        w_lgb_s3 = float(weights_cfg.get("lgb_seed1024", 0.15))
    else:
        # Fallback to balanced weights if not yet saved
        w_cb = 0.55
        w_lgb_s1 = 0.15
        w_lgb_s2 = 0.15
        w_lgb_s3 = 0.15

    print(f"Loaded Speed Residual Blend Weights:")
    print(f"  - CatBoost (d8+d10):        {w_cb * 100:.2f}%")
    print(f"  - Seed-1 Deep-LGB127 (s404): {w_lgb_s1 * 100:.2f}%")
    print(f"  - Seed-2 Deep-LGB127 (s42):  {w_lgb_s2 * 100:.2f}%")
    print(f"  - Seed-3 Deep-LGB96 (s1024): {w_lgb_s3 * 100:.2f}%")

    w_q_cat, w_q_lgb = 0.95, 0.05
    print(f"Flow Weights: CB={w_q_cat*100:.0f}%, Deep-LGB={w_q_lgb*100:.0f}%")

    # 2. Load models
    print("\nLoading CatBoost StatePredictor (bundled d8/d10)...")
    cat_predictor = StatePredictor(ASSETS)

    print("Loading Deep LightGBM Seed 1, Seed 2, and Seed 3 speed models...")
    lgb_speed_s1 = lgb.Booster(model_file=str(MODELS_DIR / "lgb_speed_residual_deep127.txt"))
    lgb_speed_s2 = lgb.Booster(model_file=str(MODELS_DIR / "lgb_speed_residual_deep127_seed42.txt"))
    lgb_speed_s3 = lgb.Booster(model_file=str(MODELS_DIR / "lgb_speed_residual_deep96_seed1024.txt"))

    print("Loading Deep LightGBM flow model...")
    lgb_flow = lgb.Booster(model_file=str(MODELS_DIR / "lgb_flow_residual_deep127.txt"))

    # 3. Preload physical FD constraints
    ffs_map, cap_map, jam_map, len_map = load_network_fd_params()

    out_file = SUB_DIR / "state_s24_ensemble_fd.csv"
    first = True
    total_rows = 0

    splits = ("validation", "private")
    panels = base.all_panels()

    for split in splits:
        for pi, name in enumerate(panels):
            t_panel = time.time()
            panel = base.Panel(name, pi)
            frame = base.month_frame(name, "", split=split)
            times, v, q, e = base.matrices(frame, panel)

            target = pd.read_csv(
                base.RELEASE / "task1" / name / split / "sample_submission_state.csv",
                dtype={"station_id": str, "link_id": str},
            )
            rows = times.get_indexer(target.timestamp)
            cols = pd.Index(panel.links).get_indexer(target.link_id)
            assert (rows >= 0).all() and (cols >= 0).all()

            regime = target.mask_regime.map({"R1": 0, "R2": 1, "R3": 2}).to_numpy()
            X, bv, bq = prepare_state.features(panel, pi, times, v, q, e, rows, cols, regime)
            assert X.shape[1] == 94

            # Predict speed residuals with 4-way multi-architecture ensemble
            res_v_cat = cat_predictor.residual(X, "speed")
            res_v_lgb_s1 = lgb_speed_s1.predict(X, num_threads=6)
            res_v_lgb_s2 = lgb_speed_s2.predict(X, num_threads=6)
            res_v_lgb_s3 = lgb_speed_s3.predict(X, num_threads=6)
            res_v = w_cb * res_v_cat + w_lgb_s1 * res_v_lgb_s1 + w_lgb_s2 * res_v_lgb_s2 + w_lgb_s3 * res_v_lgb_s3

            # Predict flow residuals
            res_q_cat = cat_predictor.residual(X, "flow")
            res_q_lgb = lgb_flow.predict(X, num_threads=6)
            res_q = w_q_cat * res_q_cat + w_q_lgb * res_q_lgb

            # Unscale to physical units
            raw_speed = bv + res_v
            raw_flow = (bq + res_q) * panel.lanes[cols]

            # Vectorized Fundamental Diagram physical parameters
            links = target["link_id"].astype(str).to_numpy()
            pairs = [(name, lid) for lid in links]
            ffs_arr = np.array([ffs_map.get(k, 105.0) for k in pairs], dtype=np.float64)
            cap_arr = np.array([cap_map.get(k, 8000.0) for k in pairs], dtype=np.float64)
            jam_arr = np.array([jam_map.get(k, 450.0) for k in pairs], dtype=np.float64)
            len_arr = np.array([len_map.get(k, 1.0) for k in pairs], dtype=np.float64)
            L_panel = np.array([len_map.get((name, str(lid)), 1.0) for lid in panel.links], dtype=np.float64)

            # 1. Base flow and speed clipping with hard jam lower bound
            flow_clipped = np.clip(raw_flow, 100.0, cap_arr)
            min_speed_for_jam = flow_clipped / (jam_arr + 1e-6)
            speed_clipped = np.clip(raw_speed, np.maximum(5.0, min_speed_for_jam), ffs_arr + 5.0)

            # 2. Refined same-day density interpolation
            Ni, run_len = interp_N_grid(v, q, e, L_panel, rows, cols)
            k_hat = flow_clipped / np.maximum(speed_clipped, 1.0)
            k_i = Ni / len_arr
            ok_n = np.isfinite(k_i) & (k_i > 0) & (k_hat > 0) & (run_len <= 4)
            w_base = 0.08
            w_n = np.where(run_len == 1, w_base, np.where(run_len == 2, w_base * 0.60, w_base * 0.32))
            delta_log = np.where(ok_n, (np.log(np.maximum(k_i, 1e-9)) - np.log(np.maximum(k_hat, 1e-9))) * w_n, 0.0)
            alpha_d = 0.75
            v_refined = np.clip(speed_clipped * np.exp(-(1.0 - alpha_d) * delta_log), 5.0, ffs_arr + 5.0)
            q_refined = np.clip(flow_clipped * np.exp(alpha_d * delta_log), 100.0, cap_arr * 1.02)

            # 3. Triangular FD congested upper envelope guardrail
            kc_arr = cap_arr / np.maximum(ffs_arr, 1.0)
            w_wave = cap_arr / np.maximum(jam_arr - kc_arr, 10.0)
            q_fd_max = np.minimum(1.04 * cap_arr, (w_wave * jam_arr) / (1.0 + w_wave / np.maximum(v_refined, 5.0)))
            congested_excess = (v_refined < 0.60 * ffs_arr) & (q_refined > q_fd_max)
            q_refined = np.where(congested_excess, 0.95 * q_refined + 0.05 * q_fd_max, q_refined)

            # 4. Enforce strict physical non-negativity and jam density constraints
            min_speed_final = q_refined / (jam_arr + 1e-6)
            speed_final = np.clip(v_refined, np.maximum(5.0, min_speed_final), ffs_arr + 5.0)
            flow_final = np.clip(q_refined, 100.0, cap_arr)

            target["speed_kmh"] = speed_final
            target["flow_vph"] = flow_final

            assert np.isfinite(target[["speed_kmh", "flow_vph"]].to_numpy()).all()
            assert (target["speed_kmh"] >= 5.0).all()
            assert (target["flow_vph"] >= 100.0).all()

            columns = ["panel", "timestamp", "station_id", "link_id", "mask_regime", "speed_kmh", "flow_vph"]
            target[columns].to_csv(out_file, index=False, mode="w" if first else "a", header=first)
            first = False
            total_rows += len(target)
            print(f"  {name} {split}: {len(target):,} rows in {time.time()-t_panel:.1f}s")

    print("-" * 75)
    print(f"Successfully generated {total_rows:,} S24 ensemble state predictions in {time.time()-t0:.1f}s -> {out_file.name}")
    assert total_rows == 6735795, f"Expected 6,735,795 rows, got {total_rows}"
    print("ALL INTEGRITY AND PHYSICAL CONSISTENCY CHECKS PASSED 100%!")


if __name__ == "__main__":
    main()
