"""Evaluate Task 3 S_FD across Convex Blend Simplex of Semwal U-Net & S24 Champion State.
Tests weights w in [0.0, 1.0] and decoupled (w_v, w_q) grids across all 10 corridors.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("trafficflowbench").resolve()
DATA = ROOT / "data" / "kaggle_public"
SEM_VAL = ROOT / "kaggle_research" / "semwal_087" / "output" / "task1" / "task1_validation.csv"
S24_STATE = ROOT / "outputs" / "submissions" / "state_s24_ensemble_fd.csv"

def network_parameters(panel_dir: Path):
    links = pd.read_csv(panel_dir / "network" / "links.csv")
    links["link_id"] = links.link_id.astype(str)
    links = links.drop_duplicates("link_id")
    keep = ["link_id", "length_km", "free_speed_kmh", "capacity_vph", "lanes"]
    out = links[keep].copy()
    for c, default in [("length_km", 1.0), ("free_speed_kmh", 100.0)]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(default)
    out["lanes"] = pd.to_numeric(out["lanes"], errors="coerce").fillna(1.0).clip(lower=1.0)
    out["capacity_vph"] = pd.to_numeric(out["capacity_vph"], errors="coerce")
    lanes = pd.to_numeric(links.get("lanes", pd.Series(4.0, index=links.index)), errors="coerce").fillna(4.0)
    lane_fallback = pd.Series(1800.0 * lanes.to_numpy(), index=links.index)
    lane_fallback.index = out.index
    out["capacity_vph"] = out.capacity_vph.fillna(lane_fallback)

    fd = panel_dir / "network" / "fd_parameters.csv"
    if fd.exists():
        f = pd.read_csv(fd)
        f["link_id"] = f.link_id.astype(str)
        have = [c for c in ("free_speed_kmh", "capacity_vph", "lanes", "k_jam") if c in f.columns]
        m = f.groupby("link_id", as_index=False)[have].mean()
        out = out.merge(m, on="link_id", how="left", suffixes=("", "_fd"))
        for c in ("free_speed_kmh", "capacity_vph", "lanes"):
            if f"{c}_fd" in out.columns:
                out[c] = out[f"{c}_fd"].fillna(out[c])
                out = out.drop(columns=[f"{c}_fd"])
        if "k_jam" not in out.columns:
            out["k_jam"] = np.nan
        out["critical_density"] = out.capacity_vph / out.free_speed_kmh.clip(lower=1)
    else:
        out["critical_density"] = out.capacity_vph / out.free_speed_kmh.clip(lower=1)
        out["k_jam"] = np.nan

    out["k_jam"] = out.k_jam.fillna(100.0 * out.lanes.clip(lower=1.0))
    out["k_jam"] = np.maximum(out.k_jam, out.critical_density * 1.05)
    return out

print("Loading Semwal validation state...")
df_sem = pd.read_csv(SEM_VAL)
print(f"Loaded Semwal: {len(df_sem):,} rows")

print("Loading S24 state...")
df_s24 = pd.read_csv(S24_STATE)
df_s24_val = df_s24[df_s24["timestamp"].str[:7] == "2031-03"].copy()
print(f"Loaded S24 val: {len(df_s24_val):,} rows")

keys = ["panel", "timestamp", "station_id", "link_id", "mask_regime"]
for k in keys:
    df_sem[k] = df_sem[k].astype(str)
    df_s24_val[k] = df_s24_val[k].astype(str)

print("Merging states...")
merged = df_sem[keys + ["speed_kmh", "flow_vph"]].merge(
    df_s24_val[keys + ["speed_kmh", "flow_vph"]],
    on=keys,
    how="inner",
    suffixes=("_sem", "_s24")
)
print(f"Aligned rows: {len(merged):,}")

# Load network parameters per panel
panels = sorted(merged["panel"].unique())
print(f"Panels ({len(panels)}): {panels}")

PANEL_PARAMS = {}
for p in panels:
    pdir = DATA / "corridors" / p
    params = network_parameters(pdir).set_index("link_id")
    PANEL_PARAMS[p] = params

def compute_s_fd_panel(df_p, params, v_col, q_col):
    x = df_p.join(params, on="link_id", how="inner")
    lanes = x.lanes.clip(lower=1.0).to_numpy()
    v = x[v_col].to_numpy()
    q = x[q_col].to_numpy()
    vf = x.free_speed_kmh.to_numpy()
    cap = x.capacity_vph.to_numpy()
    kcrit = x.critical_density.to_numpy()
    kjam = x.k_jam.to_numpy()

    q_lane = q / lanes
    k_lane = (q / np.maximum(v, 1.0)) / lanes
    cap_lane = cap / lanes
    kcrit_lane = kcrit / lanes
    kjam_lane = kjam / lanes

    q_fd = np.where(
        k_lane <= kcrit_lane,
        vf * k_lane,
        (cap_lane / np.maximum(kjam_lane - kcrit_lane, 1e-9)) * np.maximum(kjam_lane - k_lane, 0)
    )
    diff = np.abs(q_lane - q_fd)
    tot = np.abs(q_lane).sum() + 1e-9
    s_fd = float(max(0.0, 1.0 - diff.sum() / tot))
    return s_fd

def evaluate_macro_s_fd(v_blend, q_blend):
    merged["_v"] = v_blend
    merged["_q"] = q_blend
    scores = []
    for p in panels:
        p_df = merged[merged["panel"] == p]
        scores.append(compute_s_fd_panel(p_df, PANEL_PARAMS[p], "_v", "_q"))
    return np.mean(scores)

v_sem = merged["speed_kmh_sem"].to_numpy()
q_sem = merged["flow_vph_sem"].to_numpy()
v_s24 = merged["speed_kmh_s24"].to_numpy()
q_s24 = merged["flow_vph_s24"].to_numpy()

print("\n" + "=" * 70)
print(">>> TASK 3 S_FD EVALUATION OVER BLEND WEIGHT w (S24 weight) <<<")
print("=" * 70)
print(f"{'Weight w_s24':>12} | {'S_FD (macro)':>12} | {'Speed Mean':>10} | {'Flow Mean':>10}")
print("-" * 55)

weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.0]
results = []
for w in weights:
    vb = (1.0 - w) * v_sem + w * v_s24
    qb = (1.0 - w) * q_sem + w * q_s24
    s_fd = evaluate_macro_s_fd(vb, qb)
    results.append((w, s_fd, vb.mean(), qb.mean()))
    star = " <-- S33 PB" if w == 0.5 else ""
    print(f"{w:12.2f} | {s_fd:12.6f} | {vb.mean():10.2f} | {qb.mean():10.1f}{star}")

# Decoupled grid: w_v vs w_q
print("\n" + "=" * 70)
print(">>> DECOUPLED GRID SEARCH (w_v vs w_q) <<<")
print("=" * 70)
v_weights = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
q_weights = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

best_decoupled = (0.5, 0.5, 0.0)
grid_matrix = np.zeros((len(v_weights), len(q_weights)))

for i, wv in enumerate(v_weights):
    for j, wq in enumerate(q_weights):
        vb = (1.0 - wv) * v_sem + wv * v_s24
        qb = (1.0 - wq) * q_sem + wq * q_s24
        s_fd = evaluate_macro_s_fd(vb, qb)
        grid_matrix[i, j] = s_fd
        if s_fd > best_decoupled[2]:
            best_decoupled = (wv, wq, s_fd)

print(f"Optimal decoupled weights: w_v={best_decoupled[0]:.2f}, w_q={best_decoupled[1]:.2f} => S_FD = {best_decoupled[2]:.6f}")

df_grid = pd.DataFrame(grid_matrix, index=[f"wv={w:.2f}" for w in v_weights], columns=[f"wq={w:.2f}" for w in q_weights])
print("\nDecoupled Grid Table:")
print(df_grid.to_string())
