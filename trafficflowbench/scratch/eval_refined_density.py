"""Evaluate refined same-day density interpolation on February validation split.
Compares:
  - Baseline Decoupled Ensemble (70% CB + 30% Deep-LGB speed, 95% CB + 5% Deep-LGB flow)
  - With Same-Day interp_N (run-length weighted)
  - With Congested FD Upper-Envelope
Measures exact Speed RMSE, Flow/lane RMSE, S_state, and LWR conservation proxy.
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

os.environ["TFB_DATA_ROOT"] = str(DATA)
os.environ["TFB_OUTPUT_ROOT"] = str(ROOT / "scratch")
os.environ["TFB_THREADS"] = "6"

sys.path.insert(0, str(ASSETS))
import state_model as sm
from prepare_state import features
from state_inference import StatePredictor


def link_geometry(p, name):
    net = DATA / "corridors" / name / "network"
    links = pd.read_csv(net / "links.csv")
    links["link_id"] = links.link_id.astype(str)
    L = pd.to_numeric(links.drop_duplicates("link_id").set_index("link_id").length_km, errors="coerce").reindex(p.links).to_numpy(float)
    L = np.where(np.isfinite(L), L, 1.0)
    fd = pd.read_csv(net / "fd_parameters.csv")
    fd["link_id"] = fd.link_id.astype(str)
    have = [c for c in ("free_speed_kmh", "capacity_vph", "lanes", "k_jam") if c in fd]
    m = fd.groupby("link_id")[have].mean().reindex(p.links)
    lanes = m.lanes.fillna(1.0).clip(lower=1.0).to_numpy(float)
    vf = m.free_speed_kmh.to_numpy(float)
    cap = m.capacity_vph.to_numpy(float)
    kcrit = cap / np.maximum(vf, 1)
    kjam = (m.k_jam if "k_jam" in m else pd.Series(np.nan, index=m.index)).to_numpy(float)
    kjam = np.where(np.isfinite(kjam), kjam, 100.0 * lanes)
    kjam = np.maximum(kjam, 1.05 * kcrit)
    return dict(L=L, lanes=lanes, vf=vf, cap=cap, kcrit=kcrit, kjam=kjam)


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


def evaluate_postproc(v_hat, q_hat, Ni, run_len, L_cells, cap_arr, kc_arr, kj_arr, vf_arr,
                      w_base: float = 0.25, alpha: float = 0.75, max_run: int = 4, apply_fd_envelope: bool = True):
    k_hat = q_hat / np.maximum(v_hat, 1.0)
    k_i = Ni / L_cells
    ok_n = np.isfinite(k_i) & (k_i > 0) & (k_hat > 0) & (run_len <= max_run)
    
    w_n = np.where(run_len == 1, w_base, np.where(run_len == 2, w_base * 0.60, w_base * 0.32))
    delta_log = np.where(ok_n, (np.log(np.maximum(k_i, 1e-9)) - np.log(np.maximum(k_hat, 1e-9))) * w_n, 0.0)
    
    v_adj = np.clip(v_hat * np.exp(-(1.0 - alpha) * delta_log), 5.0, 140.0)
    q_adj = np.clip(q_hat * np.exp(alpha * delta_log), 100.0, 150000.0)
    
    if apply_fd_envelope:
        w_arr = cap_arr / np.maximum(kj_arr - kc_arr, 10.0)
        q_fd_max = np.minimum(1.04 * cap_arr, (w_arr * kj_arr) / (1.0 + w_arr / np.maximum(v_adj, 5.0)))
        congested_excess = (v_adj < 0.60 * vf_arr) & (q_adj > q_fd_max)
        q_adj = np.where(congested_excess, 0.95 * q_adj + 0.05 * q_fd_max, q_adj)
        
    return v_adj, q_adj


def main():
    print("=" * 70)
    print(">>> EVALUATING REFINED DENSITY INTERPOLATION ON VALIDATION BENCHMARK <<<")
    print("=" * 70)
    
    cat_predictor = StatePredictor(ASSETS)
    lgb_speed = lgb.Booster(model_file=str(MODELS_DIR / "lgb_speed_residual_deep127.txt"))
    lgb_flow = lgb.Booster(model_file=str(MODELS_DIR / "lgb_flow_residual_deep127.txt"))
    
    test_panels = ["D7_I405_N", "D7_I210_E", "D12_I5_N"]
    PROBS = []
    
    for pi, name in enumerate(test_panels):
        p = sm.Panel(name, pi)
        geo = link_geometry(p, name)
        frame = sm.month_frame(name, "2031-02", split="train")
        times, v, q, e = sm.matrices(frame, p)
        
        # 7 days of validation
        day_str = pd.to_datetime(times).strftime("%Y-%m-%d")
        unique_days = np.unique(day_str)[:7]
        sel = np.isin(day_str, unique_days)
        t_indices = np.where(sel)[0]
        
        # Simulate realistic masking across regimes
        np.random.seed(20261027)
        mask_grid = np.zeros_like(v, dtype=bool)
        for c in range(v.shape[1]):
            idx = 0
            while idx < len(t_indices):
                if np.random.rand() < 0.25:
                    run = np.random.choice([1, 2, 3, 4, 6], p=[0.45, 0.25, 0.15, 0.10, 0.05])
                    end_idx = min(idx + run, len(t_indices))
                    mask_grid[t_indices[idx:end_idx], c] = True
                    idx = end_idx + np.random.randint(2, 6)
                else:
                    idx += 1
                    
        m_rows, m_cols = np.where(mask_grid & e & np.isfinite(v) & np.isfinite(q))
        regime = np.zeros(len(m_rows), dtype=int)
        
        v_obs = np.where(mask_grid, np.nan, v)
        q_obs = np.where(mask_grid, np.nan, q)
        e_obs = np.where(mask_grid, False, e)
        
        X, bv, bq = features(p, pi, times, v_obs, q_obs, e_obs, m_rows, m_cols, regime)
        
        # S18 decoupled ensemble (70/30 speed, 95/5 flow)
        res_v_cat = cat_predictor.residual(X, "speed")
        res_v_lgb = lgb_speed.predict(X, num_threads=6)
        res_v = 0.70 * res_v_cat + 0.30 * res_v_lgb
        
        res_q_cat = cat_predictor.residual(X, "flow")
        res_q_lgb = lgb_flow.predict(X, num_threads=6)
        res_q = 0.95 * res_q_cat + 0.05 * res_q_lgb
        
        vh = bv + res_v
        qh = (bq + res_q) * p.lanes[m_cols]
        
        Ni, run_len = interp_N_grid(v_obs, q_obs, e_obs, geo["L"], m_rows, m_cols)
        
        # LWR transition ground truth
        t_step = (pd.to_datetime(times[1:]) - pd.to_datetime(times[:-1])).total_seconds() == 300
        used = e & np.isfinite(v) & np.isfinite(q)
        trans = used[:-1] & used[1:] & t_step[:, None]
        Nt = np.where(used, q / np.maximum(v, 1.0), np.nan) * geo["L"][None, :]
        
        PROBS.append(dict(name=name, p=p, geo=geo, rows=m_rows, cols=m_cols,
                          v_true=v[m_rows, m_cols], q_true=q[m_rows, m_cols],
                          vh=vh, qh=qh, Ni=Ni, run_len=run_len, trans=trans, Nt=Nt,
                          v_obs=v_obs, q_obs=q_obs))
        print(f"Loaded {name}: {len(m_rows):,} simulated masked cells, anchored = {np.isfinite(Ni).mean():.1%}")

    def score_all(v_preds, q_preds):
        rv_list, rq_list, lwr_list = [], [], []
        for P, vp, qp in zip(PROBS, v_preds, q_preds):
            lanes = P["geo"]["lanes"][P["cols"]]
            rv = np.sqrt(np.mean((vp - P["v_true"]) ** 2))
            rq = np.sqrt(np.mean(((qp - P["q_true"]) / lanes) ** 2))
            
            # LWR conservation proxy
            V_full = P["v_obs"].copy()
            Q_full = P["q_obs"].copy()
            V_full[P["rows"], P["cols"]] = vp
            Q_full[P["rows"], P["cols"]] = qp
            N = (Q_full / np.maximum(V_full, 1.0)) * P["geo"]["L"][None, :]
            dN = N[1:] - N[:-1]
            dNt = P["Nt"][1:] - P["Nt"][:-1]
            tr = P["trans"]
            s_lwr = max(0.0, 1.0 - float(np.sum(np.abs(dN - dNt)[tr])) / max(float(np.sum(np.abs(dNt)[tr])), 1.0))
            
            rv_list.append(rv)
            rq_list.append(rq)
            lwr_list.append(s_lwr)
            
        mean_rv = float(np.mean(rv_list))
        mean_rq = float(np.mean(rq_list))
        mean_lwr = float(np.mean(lwr_list))
        s_state = 0.54 * max(0, 1 - mean_rv / 25.0) + 0.46 * max(0, 1 - mean_rq / 600.0)
        s_physics_lwr = mean_lwr  # LWR is 2/3 of Task 3
        # Composite Task 1 + Task 3 gain
        est_score = 0.35 * s_state + 0.15 * (0.95 / 3.0 + 2.0 * s_physics_lwr / 3.0)
        return dict(rv=mean_rv, rq=mean_rq, s_state=s_state, s_lwr=mean_lwr, est_score=est_score)

    # 1. Baseline S18 Decoupled predictions
    base_res = score_all([P["vh"] for P in PROBS], [P["qh"] for P in PROBS])
    print("\n--- BASELINE (S18 Decoupled 70/30 Speed + 95/5 Flow) ---")
    print(f"Speed RMSE:     {base_res['rv']:.4f} km/h")
    print(f"Flow/lane RMSE: {base_res['rq']:.4f} vph/lane")
    print(f"S_state:        {base_res['s_state']:.5f}")
    print(f"LWR Proxy:      {base_res['s_lwr']:.5f}")
    print(f"Estimated LB:   {base_res['est_score']:.5f}")

    # 2. Test grid of density adjustment
    print("\n--- GRID OVER SAME-DAY REFINEMENT ---")
    print(f"{'w_base':<8} | {'alpha':<6} | {'max_run':<7} | {'FD_env':<7} | {'Speed RMSE':<11} | {'Flow RMSE':<11} | {'LWR':<8} | {'Est. LB':<9} | {'Delta LB':<9}")
    print("-" * 88)
    
    best_delta = -999.0
    best_params = None

    for apply_fd in [True, False]:
        for w_base in [0.05, 0.10, 0.15, 0.20, 0.25]:
            for alpha in [0.75, 0.85, 0.95, 1.00]:
                for max_run in [2, 4]:
                    v_adj_list, q_adj_list = [], []
                    for P in PROBS:
                        geo = P["geo"]
                        cols = P["cols"]
                        va, qa = evaluate_postproc(
                            P["vh"], P["qh"], P["Ni"], P["run_len"], geo["L"][cols],
                            geo["cap"][cols], geo["kcrit"][cols], geo["kjam"][cols], geo["vf"][cols],
                            w_base=w_base, alpha=alpha, max_run=max_run, apply_fd_envelope=apply_fd
                        )
                        v_adj_list.append(va)
                        q_adj_list.append(qa)
                    
                    res = score_all(v_adj_list, q_adj_list)
                    delta = res["est_score"] - base_res["est_score"]
                    marker = " *** BEST ***" if delta > best_delta else ""
                    if delta > best_delta:
                        best_delta = delta
                        best_params = (w_base, alpha, max_run, apply_fd)
                    print(f"{w_base:<8.2f} | {alpha:<6.2f} | {max_run:<7d} | {str(apply_fd):<7} | {res['rv']:<11.4f} | {res['rq']:<11.4f} | {res['s_lwr']:<8.4f} | {res['est_score']:<9.5f} | {delta:<+9.5f}{marker}")

    print("\n" + "=" * 70)
    print(f"OPTIMAL REFINEMENT: w={best_params[0]}, alpha={best_params[1]}, max_run={best_params[2]}, FD={best_params[3]}")
    print(f"ESTIMATED LEADERBOARD GAIN: {best_delta:+.5f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
