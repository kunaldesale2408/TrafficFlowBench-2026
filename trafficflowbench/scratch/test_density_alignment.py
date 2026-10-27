"""Test log-space density alignment on February validation for D7_I405_N.
Measures:
  - Speed RMSE
  - Flow RMSE
  - S_state
  - S_LWR
  - S_FD
  - Estimated composite LB gain
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

def main():
    print("=" * 60)
    print("Testing Density Post-Processing on Validation Corridor")
    print("=" * 60)
    
    panel_name = "D7_I405_N"
    p = sm.Panel(panel_name, 4)
    geo = link_geometry(p, panel_name)
    
    cat_predictor = StatePredictor(ASSETS)
    lgb_speed = lgb.Booster(model_file=str(MODELS_DIR / "lgb_speed_residual_94f.txt"))
    lgb_flow = lgb.Booster(model_file=str(MODELS_DIR / "lgb_flow_residual_94f.txt"))
    
    # Load 7 days from validation split
    # Truth is in train continuous data or month_frame
    train_frame = sm.month_frame(panel_name, "2031-02", split="train")
    times, v_true, q_true, e_true = sm.matrices(train_frame, p)
    
    # Read validation masked frame to see how masks look
    val_frame = sm.month_frame(panel_name, "", split="validation")
    val_times, v_obs, q_obs, e_obs = sm.matrices(val_frame, p)
    
    # On train, simulate masked cells
    days = np.unique(pd.to_datetime(times).strftime("%Y-%m-%d"))[:7]
    mask_day = np.isin(pd.to_datetime(times).strftime("%Y-%m-%d"), days)
    
    # Use real masked target from validation template to test interpolation
    target = pd.read_csv(
        DATA / "task1" / panel_name / "validation" / "sample_submission_state.csv",
        dtype={"station_id": str, "link_id": str},
    )
    rows = val_times.get_indexer(target.timestamp)
    cols = pd.Index(p.links).get_indexer(target.link_id)
    regime = target.mask_regime.map({"R1": 0, "R2": 1, "R3": 2}).to_numpy()
    
    print(f"Validation targets: {len(target):,} cells")
    Ni, run = interp_N(v_obs, q_obs, e_obs, geo["L"], rows, cols)
    print(f"Anchored with known observations: {np.isfinite(Ni).mean():.1%}")
    print(f"Run length distribution: min={run.min()}, median={np.median(run[run<1000])}, max={run[run<1000].max()}")
    print("Density interpolation ready.")

if __name__ == "__main__":
    main()
