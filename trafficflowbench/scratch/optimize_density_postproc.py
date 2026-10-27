"""Optimize density post-processing hyperparameters on February held-out days.
Finds exact w, alpha, max_run that maximizes 0.35 * S_state + 0.15 * S_physics.
"""
from __future__ import annotations

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

def interp_N(v_obs, q_obs, e, L, rows, cols):
    ok = e & np.isfinite(v_obs) & np.isfinite(q_obs)
    N = np.where(ok, q_obs / np.maximum(v_obs, 1.0) * L[None, :], np.nan)
    T = N.shape[0]
    pos = np.arange(T)[:, None]
    prev = np.maximum.accumulate(np.where(ok, pos, -1), axis=0)
    nxt = np.minimum.accumulate(np.where(ok, pos, T)[::-1], axis=0)[::-1]
    p, n = prev[rows, cols], nxt[rows, cols]
    has = (p >= 0) & (n < T)
    pp, nn = np.maximum(p, 0), np.minimum(n, T - 1)
    t = (rows - pp) / np.maximum(nn - pp, 1)
    Ni = N[pp, cols] * (1 - t) + N[nn, cols] * t
    return np.where(has, Ni, np.nan), np.where(has, nn - pp - 1, 10**6)

def adjust(v_hat, q_hat, Ni, run, L_cells, w, alpha, max_run):
    k_hat = q_hat / np.maximum(v_hat, 1.0)
    k_i = Ni / L_cells
    ok = np.isfinite(k_i) & (k_i > 0) & (k_hat > 0) & (run <= max_run)
    d = np.where(ok, np.log(np.maximum(k_i, 1e-9)) - np.log(np.maximum(k_hat, 1e-9)), 0.0) * w
    return np.clip(v_hat * np.exp(-(1 - alpha) * d), 5.0, 150.0), np.clip(q_hat * np.exp(alpha * d), 100.0, 150000.0)

def scores(P, vn, qn):
    g, rows, cols = P["geo"], P["rows"], P["cols"]
    V, Q = P["v"].copy(), P["q"].copy()
    V[rows, cols] = vn
    Q[rows, cols] = qn
    N = Q / np.maximum(V, 1.0) * g["L"][None, :]
    Nt = P["Nt"]
    dN, dNt = N[1:] - N[:-1], Nt[1:] - Nt[:-1]
    lanes = g["lanes"][None, :]
    ql, kl = Q / lanes, Q / np.maximum(V, 1.0) / lanes
    kc, kj, cp = (g["kcrit"] / g["lanes"])[None, :], (g["kjam"] / g["lanes"])[None, :], (g["cap"] / g["lanes"])[None, :]
    qfd = np.where(kl <= kc, g["vf"][None, :] * kl, cp / np.maximum(kj - kc, 1e-9) * np.maximum(kj - kl, 0))
    
    # Official evaluation metrics
    rv = np.sqrt(np.mean((vn - P["vt"]) ** 2))
    rq = np.sqrt(np.mean(((qn - P["qt"]) / g["lanes"][cols]) ** 2))
    s_state = 0.5 * np.exp(-rv / 25.0) + 0.5 * np.exp(-rq / 600.0)
    
    # LWR conservation proxy (transitions on evaluated cells)
    tr = P["trans"]
    res = np.abs(dN - dNt)[tr]
    denom = np.abs(dNt)[tr]
    s_lwr = max(0.0, 1.0 - float(np.sum(res)) / max(float(np.sum(denom)), 1.0))
    
    # FD physical penalty
    fd_err = np.mean(np.maximum(0, ql[rows, cols] - qfd[rows, cols]))
    s_fd = max(0.0, 1.0 - fd_err / 500.0)
    
    return dict(rv=rv, rq=rq, s_state=s_state, s_lwr=s_lwr, s_fd=s_fd,
                composite=0.35 * s_state + 0.15 * (s_fd / 3.0 + 2.0 * s_lwr / 3.0))

def main():
    cat_predictor = StatePredictor(ASSETS)
    lgb_speed = lgb.Booster(model_file=str(MODELS_DIR / "lgb_speed_residual_94f.txt"))
    lgb_flow = lgb.Booster(model_file=str(MODELS_DIR / "lgb_flow_residual_94f.txt"))
    
    test_panels = ["D7_I405_N", "D7_I210_E"]
    PROBS = []
    
    for pi, name in enumerate(test_panels):
        p = sm.Panel(name, pi)
        geo = link_geometry(p, name)
        frame = sm.month_frame(name, "2031-02", split="train")
        times, v, q, e = sm.matrices(frame, p)
        
        # Take 5 days of masked cells
        day_str = pd.to_datetime(times).strftime("%Y-%m-%d")
        unique_days = np.unique(day_str)[:5]
        sel = np.isin(day_str, unique_days)
        
        # Artificially mask 25% of cells like R1/R2
        np.random.seed(42)
        n_rows, n_cols = v.shape
        t_indices = np.where(sel)[0]
        
        # Simulate realistic runs of missing cells
        mask_grid = np.zeros_like(v, dtype=bool)
        for c in range(n_cols):
            idx = 0
            while idx < len(t_indices):
                if np.random.rand() < 0.20:
                    run = np.random.choice([1, 2, 3, 4], p=[0.5, 0.25, 0.15, 0.10])
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
        
        # Extract features
        X, bv, bq = features(p, pi, times, v_obs, q_obs, e_obs, m_rows, m_cols, regime)
        
        # Decoupled prediction (85/15 speed, 95/5 flow)
        res_v = 0.85 * cat_predictor.residual(X, "speed") + 0.15 * lgb_speed.predict(X, num_threads=4)
        res_q = 0.95 * cat_predictor.residual(X, "flow") + 0.05 * lgb_flow.predict(X, num_threads=4)
        
        vh = bv + res_v
        qh = (bq + res_q) * p.lanes[m_cols]
        
        Ni, run = interp_N(v_obs, q_obs, e_obs, geo["L"], m_rows, m_cols)
        
        t_step = (pd.to_datetime(times[1:]) - pd.to_datetime(times[:-1])).total_seconds() == 300
        used = e & np.isfinite(v) & np.isfinite(q)
        trans = used[:-1] & used[1:] & t_step[:, None]
        Nt = np.where(used, q / np.maximum(v, 1.0), np.nan) * geo["L"][None, :]
        
        PROBS.append(dict(name=name, geo=geo, rows=m_rows, cols=m_cols, v=v_obs, q=q_obs,
                          vt=v[m_rows, m_cols], qt=q[m_rows, m_cols], vh=vh, qh=qh,
                          Ni=Ni, run=run, trans=trans, Nt=Nt))
        print(f"Loaded {name}: {len(m_rows):,} simulated masked cells")

    # Baseline scores (no adjustment)
    base_scores = [scores(P, P["vh"], P["qh"]) for P in PROBS]
    base_comp = np.mean([s["composite"] for s in base_scores])
    print(f"\nBASELINE (Decoupled Ensemble): Composite = {base_comp:.5f} (State = {np.mean([s['s_state'] for s in base_scores]):.5f}, LWR = {np.mean([s['s_lwr'] for s in base_scores]):.5f})")

    # Grid search over w, alpha, max_run
    grid_w = [0.10, 0.15, 0.20, 0.25, 0.30]
    grid_alpha = [0.70, 0.75, 0.80, 0.85]
    grid_run = [2, 3, 4]
    
    best_gain = -999.0
    best_cfg = None
    
    for mr in grid_run:
        for a in grid_alpha:
            for w in grid_w:
                sc_list = []
                for P in PROBS:
                    vn, qn = adjust(P["vh"], P["qh"], P["Ni"], P["run"], P["geo"]["L"][P["cols"]], w, a, mr)
                    sc_list.append(scores(P, vn, qn))
                comp = np.mean([s["composite"] for s in sc_list])
                gain = comp - base_comp
                if gain > best_gain:
                    best_gain = gain
                    best_cfg = (w, a, mr)
                    print(f"  w={w:.2f}, alpha={a:.2f}, max_run={mr} -> Comp={comp:.5f} (Gain={gain:+.5f}, State={np.mean([s['s_state'] for s in sc_list]):.5f}, LWR={np.mean([s['s_lwr'] for s in sc_list]):.5f})")

    print("\n" + "=" * 60)
    print(f"OPTIMAL DENSITY REFINEMENT: w={best_cfg[0]}, alpha={best_cfg[1]}, max_run={best_cfg[2]} -> Gain: {best_gain:+.5f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
