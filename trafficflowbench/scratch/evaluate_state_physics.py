"""Benchmark LWR Conservation Proxy & Task 1 State Score on February 2031 across corridors.
Measures the true impact of density post-processing (interp_N weight w, alpha, max_run)
on Task 1 S_state, Task 3 S_FD, Task 3 S_LWR, and composite S_T1_T3.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path("trafficflowbench").resolve()
DATA = ROOT / "data" / "kaggle_public"
ASSETS = ROOT / "kaggle_research" / "anubis_assets"
MODELS_DIR = ROOT / "outputs" / "models"

os.environ["TFB_DATA_ROOT"] = str(DATA)
os.environ["TFB_OUTPUT_ROOT"] = str(ROOT / "scratch")
os.environ["TFB_THREADS"] = "6"
os.environ["TFB_PREDICT_DEVICE"] = "CPU"

sys.path.insert(0, str(ASSETS))
import state_model as base
import prepare_state


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


def interp_N_grid(v_obs, q_obs, e, L, rows, cols):
    ok = e & np.isfinite(v_obs) & np.isfinite(q_obs) & (v_obs > 1.0)
    N = np.where(ok, q_obs / np.maximum(v_obs, 1.0) * L[None, :], np.nan)
    T = N.shape[0]
    pos = np.arange(T, dtype=np.int32)[:, None]
    prev = np.maximum.accumulate(np.where(ok, pos, -1), axis=0)
    nxt = np.minimum.accumulate(np.where(ok, pos, T)[::-1], axis=0)[::-1]
    p, n = prev[rows, cols], nxt[rows, cols]
    has = (p >= 0) & (n < T) & ((rows // 288) == (np.maximum(p, 0) // 288)) & ((rows // 288) == (np.minimum(n, T - 1) // 288))
    pp, nn = np.maximum(p, 0), np.minimum(n, T - 1)
    t = (rows - pp) / np.maximum(nn - pp, 1)
    Ni = N[pp, cols] * (1 - t) + N[nn, cols] * t
    return np.where(has, Ni, np.nan), np.where(has, nn - pp - 1, 10**6)


def adjust(v_hat, q_hat, Ni, run, L_cells, w, alpha, max_run):
    k_hat = q_hat / np.maximum(v_hat, 1.0)
    k_i = Ni / L_cells
    ok = np.isfinite(k_i) & (k_i > 0) & (k_hat > 0) & (run <= max_run)
    w_effective = np.where(run == 1, w, np.where(run == 2, w * 0.70, w * 0.40))
    d = np.where(ok, np.log(np.maximum(k_i, 1e-9)) - np.log(np.maximum(k_hat, 1e-9)), 0.0) * w_effective
    return np.clip(v_hat * np.exp(-(1 - alpha) * d), 5.0, 150.0), np.clip(q_hat * np.exp(alpha * d), 100.0, 150000.0)


def evaluate_setup():
    print("=" * 75)
    print(">>> EVALUATING TASK 1 & TASK 3 PHYSICS OPTIMIZATION (FEBRUARY HELD-OUT) <<<")
    print("=" * 75)

    print("Loading models...")
    cb_d8 = CatBoostRegressor()
    cb_d8.load_model(str(ASSETS / "speed_d8.cbm"))
    cb_d10 = CatBoostRegressor()
    cb_d10.load_model(str(ASSETS / "speed_d10.cbm"))
    lgb_speed_s1 = lgb.Booster(model_file=str(MODELS_DIR / "lgb_speed_residual_deep127.txt"))
    lgb_speed_s2 = lgb.Booster(model_file=str(MODELS_DIR / "lgb_speed_residual_deep127_seed42.txt"))
    lgb_flow = lgb.Booster(model_file=str(MODELS_DIR / "lgb_flow_residual_deep127.txt"))
    cb_q_d8 = CatBoostRegressor()
    cb_q_d8.load_model(str(ASSETS / "flow_d8.cbm"))
    cb_q_d10 = CatBoostRegressor()
    cb_q_d10.load_model(str(ASSETS / "flow_d10.cbm"))

    # S24 Speed Weights
    w_cb = 0.57556
    w_m1 = 0.06731
    w_m2 = 0.35713

    # Corridors to evaluate
    eval_panels = ["D7_I10_E", "D7_I210_E", "D7_I405_N", "D12_I5_N"]
    PROBS = []

    for pi, name in enumerate(eval_panels):
        t_p = time.time()
        p = base.Panel(name, pi)
        geo = link_geometry(p, name)
        
        # Load 7 days from Feb 2031
        frame = base.month_frame(name, "2031-02")
        frame["timestamp"] = frame.timestamp.astype(str)
        unique_days = sorted(frame.timestamp.str[:10].unique())[:7]
        frame = frame[frame.timestamp.str[:10].isin(unique_days)]
        times, v, q, e = base.matrices(frame, p)
        
        mask, regime = base.simulated_mask(name, times, v, q, e)
        rows, cols = np.nonzero(mask & e & np.isfinite(v) & np.isfinite(q))
        
        # Build features
        v_obs = np.where(mask, np.nan, v)
        q_obs = np.where(mask, np.nan, q)
        e_obs = np.where(mask, False, e)
        reg_cells = regime[rows]
        
        X, bv, bq = prepare_state.features(p, pi, times, v_obs, q_obs, e_obs, rows, cols, reg_cells)
        
        # S24 Speed residual
        p_cb_v = 0.5 * cb_d8.predict(X, thread_count=6) + 0.5 * cb_d10.predict(X, thread_count=6)
        p_lgb_v = w_m1 * lgb_speed_s1.predict(X, num_threads=6) + w_m2 * lgb_speed_s2.predict(X, num_threads=6)
        res_v = w_cb * p_cb_v + p_lgb_v
        
        # Flow residual (95/5)
        p_cb_q = 0.5 * cb_q_d8.predict(X, thread_count=6) + 0.5 * cb_q_d10.predict(X, thread_count=6)
        p_lgb_q = lgb_flow.predict(X, num_threads=6)
        res_q = 0.95 * p_cb_q + 0.05 * p_lgb_q
        
        vh = bv + res_v
        qh = (bq + res_q) * p.lanes[cols]
        
        # Base clipping
        cap_arr = geo["cap"][cols]
        ffs_arr = geo["vf"][cols]
        jam_arr = geo["kjam"][cols]
        qh = np.clip(qh, 100.0, cap_arr)
        min_v = qh / (jam_arr + 1e-6)
        vh = np.clip(vh, np.maximum(5.0, min_v), ffs_arr + 5.0)
        
        Ni, run = interp_N_grid(v_obs, q_obs, e_obs, geo["L"], rows, cols)
        
        t_step = (pd.to_datetime(times[1:]) - pd.to_datetime(times[:-1])).total_seconds() == 300
        used = e & np.isfinite(v) & np.isfinite(q)
        trans = used[:-1] & used[1:] & t_step[:, None]
        Nt = np.where(used, q / np.maximum(v, 1.0), np.nan) * geo["L"][None, :]
        
        PROBS.append(dict(
            name=name, p=p, geo=geo, rows=rows, cols=cols, reg_cells=reg_cells,
            v=v_obs, q=q_obs, vt=v[rows, cols], qt=q[rows, cols], vh=vh, qh=qh,
            Ni=Ni, run=run, trans=trans, Nt=Nt, used=used, reg_row=regime
        ))
        print(f"Loaded {name}: {len(rows):,} masked cells ({np.isfinite(Ni).mean():.1%} anchored) in {time.time()-t_p:.1f}s")
        
    return PROBS


def compute_scores(PROBS, w, alpha, max_run):
    res_list = []
    for P in PROBS:
        if w == 0.0:
            vn, qn = P["vh"], P["qh"]
        else:
            vn, qn = adjust(P["vh"], P["qh"], P["Ni"], P["run"], P["geo"]["L"][P["cols"]], w, alpha, max_run)
            
        g, rows, cols = P["geo"], P["rows"], P["cols"]
        V, Q = P["v"].copy(), P["q"].copy()
        V[rows, cols] = vn
        Q[rows, cols] = qn
        N = Q / np.maximum(V, 1.0) * g["L"][None, :]
        Nt = P["Nt"]
        dN, dNt = N[1:] - N[:-1], Nt[1:] - Nt[:-1]
        
        lanes = g["lanes"][None, :]
        ql = Q / lanes
        kl = Q / np.maximum(V, 1.0) / lanes
        kc = (g["kcrit"] / g["lanes"])[None, :]
        kj = (g["kjam"] / g["lanes"])[None, :]
        cp = (g["cap"] / g["lanes"])[None, :]
        qfd = np.where(kl <= kc, g["vf"][None, :] * kl, cp / np.maximum(kj - kc, 1e-9) * np.maximum(kj - kl, 0))
        
        rv = np.sqrt(np.mean((vn - P["vt"]) ** 2))
        rq = np.sqrt(np.mean(((qn - P["qt"]) / g["lanes"][cols]) ** 2))
        s_state = 0.54 * max(0.0, 1.0 - rv / 25.0) + 0.46 * max(0.0, 1.0 - rq / 600.0)
        
        tr = P["trans"]
        s_lwr = max(0.0, 1.0 - float(np.sum(np.abs(dN - dNt)[tr])) / max(float(np.sum(np.abs(dNt)[tr])), 1e-9))
        
        u = P["used"]
        s_fd = max(0.0, 1.0 - float(np.sum(np.abs(ql - qfd)[u])) / max(float(np.sum(np.abs(ql)[u])), 1e-9))
        
        s_phys = (s_fd + 2.0 * s_lwr) / 3.0
        # Total contribution from Task 1 (35%) + Task 3 (15%)
        # Note: Task 2 queue and Task 4 ODME are fixed
        sub_contrib = 0.35 * s_state + 0.15 * s_phys
        
        res_list.append(dict(rv=rv, rq=rq, s_state=s_state, s_lwr=s_lwr, s_fd=s_fd, s_phys=s_phys, sub_contrib=sub_contrib))
        
    df = pd.DataFrame(res_list).mean()
    return df


def main():
    probs = evaluate_setup()
    
    # 1. Baseline (w=0, raw state predictions)
    b0 = compute_scores(probs, w=0.0, alpha=0.75, max_run=4)
    print("\n--- BASELINE (Raw Bagged Deep-LGB State, w=0.0) ---")
    print(f"Speed RMSE:     {b0['rv']:.4f} km/h")
    print(f"Flow/lane RMSE: {b0['rq']:.4f} vph/lane")
    print(f"S_state:        {b0['s_state']:.5f}")
    print(f"S_FD:           {b0['s_fd']:.5f}")
    print(f"S_LWR:          {b0['s_lwr']:.5f}")
    print(f"S_physics:      {b0['s_phys']:.5f}")
    print(f"T1+T3 Contrib:  {b0['sub_contrib']:.5f}")
    
    # 2. S24 setting (w=0.08)
    b_s24 = compute_scores(probs, w=0.08, alpha=0.75, max_run=4)
    print("\n--- CURRENT PB S24 SETTING (w=0.08, alpha=0.75, max_run=4) ---")
    print(f"Speed RMSE:     {b_s24['rv']:.4f} km/h (Delta = {b_s24['rv']-b0['rv']:+.4f})")
    print(f"Flow/lane RMSE: {b_s24['rq']:.4f} vph/lane (Delta = {b_s24['rq']-b0['rq']:+.4f})")
    print(f"S_state:        {b_s24['s_state']:.5f} (Delta = {b_s24['s_state']-b0['s_state']:+.5f})")
    print(f"S_LWR:          {b_s24['s_lwr']:.5f} (Delta = {b_s24['s_lwr']-b0['s_lwr']:+.5f})")
    print(f"T1+T3 Contrib:  {b_s24['sub_contrib']:.5f} (Delta = {b_s24['sub_contrib']-b0['sub_contrib']:+.5f})")

    # 3. Grid search over w and alpha
    print("\n" + "=" * 80)
    print(f"{'w':<6} | {'alpha':<6} | {'max_run':<7} | {'Speed RMSE':<11} | {'Flow RMSE':<11} | {'S_state':<8} | {'S_LWR':<8} | {'S_phys':<8} | {'Contrib':<9} | {'LB Gain':<9}")
    print("-" * 80)
    
    best_gain = -999.0
    best_cfg = None
    
    for mr in [2, 4, 6]:
        for a in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
            for w in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.45, 0.60]:
                sc = compute_scores(probs, w=w, alpha=a, max_run=mr)
                gain = sc["sub_contrib"] - b_s24["sub_contrib"]
                marker = " *** BEST ***" if gain > best_gain else ""
                if gain > best_gain:
                    best_gain = gain
                    best_cfg = (w, a, mr, sc)
                print(f"{w:<6.2f} | {a:<6.2f} | {mr:<7d} | {sc['rv']:<11.4f} | {sc['rq']:<11.4f} | {sc['s_state']:<8.4f} | {sc['s_lwr']:<8.4f} | {sc['s_phys']:<8.4f} | {sc['sub_contrib']:<9.5f} | {gain:<+9.5f}{marker}")

    print("=" * 80)
    print(f"OPTIMAL CONFIGURATION: w={best_cfg[0]}, alpha={best_cfg[1]}, max_run={best_cfg[2]}")
    print(f"ESTIMATED LEADERBOARD SCORE GAIN OVER PB S24: {best_gain:+.5f} points!")
    print(f"PROJECTED LEADERBOARD SCORE: {0.86760 + best_gain:.5f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
