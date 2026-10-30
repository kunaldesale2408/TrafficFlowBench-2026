"""Evaluate Ongoing Queue Blends and Thresholds on Train Ground Truth.
Tests:
  - LGB pure vs 88/12 LGB/CB vs 80/20 vs CB pure
  - Threshold sweeps (0.40 to 0.55)
  - District breakdowns (D7 vs D12)
"""
from __future__ import annotations

import os
os.environ["TFB_DATA_ROOT"] = "trafficflowbench/data/kaggle_public"
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path("trafficflowbench").resolve()
DATA = ROOT / "data" / "kaggle_public"
ASSETS = ROOT / "kaggle_research" / "anubis_assets"
MODELS = ROOT / "outputs" / "models"

sys.path.insert(0, str(ASSETS))
import queue_ongoing as q_ongoing_module
from queue_features import panel_names, HISTORY, HORIZON

# 1. Load models
m_ong_lgb = lgb.Booster(model_file=str(ASSETS / "ongoing_lgb.txt"))
m_ong_cb = CatBoostClassifier()
m_ong_cb.load_model(str(MODELS / "ongoing_cb.cbm"))

# 2. Load free speed per link
ffs = {}
for p in (DATA / "corridors").iterdir():
    if p.is_dir():
        fd = p / "network" / "fd_parameters.csv"
        if fd.exists():
            df = pd.read_csv(fd, dtype={"link_id": str})
            for _, r in df.iterrows():
                ffs[(p.name, str(r["link_id"]))] = float(r.get("free_speed_kmh", 105.0))

print("Extracting predictions and ground truth for all train ongoing windows...")
t0 = time.time()
window_records = []

for panel_no, name in enumerate(panel_names()):
    panel_ong = q_ongoing_module.Panel(name, panel_no)

    w_file = DATA / "task2" / name / "train" / "window_index.csv"
    if not w_file.exists():
        continue
    windows = pd.read_csv(w_file, dtype={"window_id": str})
    ongoing = windows[windows.condition == "queue_ongoing"]
    history = pd.read_parquet(DATA / "task2" / name / "train" / "window_history.parquet")

    # Read unmasked state for ground truth
    state_files = sorted((DATA / "corridors" / name / "train" / "mainline_states").glob("**/*.parquet"))
    s_dfs = [pd.read_parquet(sf, columns=["timestamp", "link_id", "speed_kmh"]) for sf in state_files]
    state_df = pd.concat(s_dfs, ignore_index=True)
    state_df["timestamp"] = pd.to_datetime(state_df["timestamp"], utc=True)
    state_df["link_id"] = state_df["link_id"].astype(str)

    for row in ongoing.itertuples(index=False):
        origin = pd.Timestamp(row.forecast_origin)
        times = pd.date_range(end=origin - pd.Timedelta(minutes=5), periods=HISTORY, freq="5min", tz="UTC")
        index = pd.Index(times.strftime("%Y-%m-%dT%H:%M:%SZ"))
        h = history[history.window_id.astype(str) == str(row.window_id)].copy()
        h["link_id"] = h.link_id.astype(str)

        sp = h.pivot(index="timestamp", columns="link_id", values="speed_kmh").reindex(index=index, columns=panel_ong.links).to_numpy(dtype=np.float32)
        fl = h.pivot(index="timestamp", columns="link_id", values="flow_vph").reindex(index=index, columns=panel_ong.links).to_numpy(dtype=np.float32)
        observed = h.pivot(index="timestamp", columns="link_id", values="is_score_eligible").reindex(index=index, columns=panel_ong.links).fillna(0).to_numpy(dtype=bool)

        X = panel_ong.features(sp, fl, observed, origin, "queue_ongoing")
        p_lgb = m_ong_lgb.predict(X, num_threads=6)
        p_cb = m_ong_cb.predict_proba(X, thread_count=6)[:, 1]

        # Ground truth for all 6 horizons: T+5, T+10, ..., T+30
        y_true_list = []
        for step in range(1, HORIZON + 1):
            t_step = origin + pd.Timedelta(minutes=5 * step)
            snap = state_df[state_df.timestamp == t_step]
            step_y = np.zeros(len(panel_ong.links), dtype=bool)
            if not snap.empty:
                speed_map = dict(zip(snap["link_id"], snap["speed_kmh"]))
                for li, lid in enumerate(panel_ong.links):
                    v_true = speed_map.get(str(lid), np.nan)
                    v_cut = 0.60 * ffs.get((name, str(lid)), 105.0)
                    if np.isfinite(v_true) and v_true <= v_cut:
                        step_y[li] = True
            y_true_list.append(step_y)

        y_true = np.concatenate(y_true_list)

        window_records.append({
            "panel": name,
            "district": name.split("_")[0],
            "window_id": str(row.window_id),
            "p_lgb": p_lgb,
            "p_cb": p_cb,
            "y_true": y_true,
            "n_cells": len(y_true),
        })

print(f"Extracted data for {len(window_records)} ongoing windows in {time.time()-t0:.1f}s.")

# Evaluate IoU function
def calc_iou(preds, truths):
    # Mean per-window IoU
    ious = []
    for p, y in zip(preds, truths):
        intersection = (p & y).sum()
        union = (p | y).sum()
        if union == 0:
            ious.append(1.0 if intersection == 0 else 0.0)
        else:
            ious.append(intersection / union)
    return float(np.mean(ious))

# Evaluate blend weights and thresholds
results = []
thresholds = [0.40, 0.42, 0.44, 0.45, 0.46, 0.47, 0.48, 0.485, 0.49, 0.495, 0.50, 0.52]
weights = [1.00, 0.88, 0.80, 0.50, 0.00]

for w in weights:
    for th in thresholds:
        preds = []
        truths = []
        preds_d7, truths_d7 = [], []
        preds_d12, truths_d12 = [], []

        for rec in window_records:
            p_blend = w * rec["p_lgb"] + (1.0 - w) * rec["p_cb"]
            pred = p_blend >= th
            preds.append(pred)
            truths.append(rec["y_true"])

            if rec["district"] == "D7":
                preds_d7.append(pred)
                truths_d7.append(rec["y_true"])
            else:
                preds_d12.append(pred)
                truths_d12.append(rec["y_true"])

        iou_overall = calc_iou(preds, truths)
        iou_d7 = calc_iou(preds_d7, truths_d7)
        iou_d12 = calc_iou(preds_d12, truths_d12)

        results.append({
            "w_lgb": w,
            "th": th,
            "iou_overall": iou_overall,
            "iou_d7": iou_d7,
            "iou_d12": iou_d12,
        })

res_df = pd.DataFrame(results)
print("\n" + "=" * 65)
print(">>> TOP 15 ONGOING BLEND & THRESHOLD CONFIGURATIONS <<<")
print("=" * 65)
print(res_df.sort_values("iou_overall", ascending=False).head(15).to_string(index=False))

print("\n" + "=" * 65)
print(">>> PURE LGB (w=1.00) vs S24 BLEND (w=0.88) by THRESHOLD <<<")
print("=" * 65)
comp = res_df[res_df.w_lgb.isin([1.00, 0.88])].pivot(index="th", columns="w_lgb", values=["iou_overall", "iou_d7", "iou_d12"])
print(comp.round(4).to_string())
