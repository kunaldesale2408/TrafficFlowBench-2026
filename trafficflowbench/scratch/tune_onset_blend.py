"""Tune Onset Ensemble Blend Weights (LightGBM vs CatBoost) on Train Ground Truth Windows.
Evaluates IoU across different model blend weights:
  p = w * p_lgb + (1 - w) * p_cb
and decision thresholds tau.
"""
from __future__ import annotations

import os
os.environ['TFB_DATA_ROOT'] = 'trafficflowbench/data/kaggle_public'
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
PRIORS_DIR = ROOT / "outputs" / "priors"

sys.path.insert(0, str(ASSETS))
import queue_features as q_onset_module
from queue_features import panel_names, HISTORY, HORIZON

# 1. Load models
m_on_lgb = lgb.Booster(model_file=str(ASSETS / "onset_lgb.txt"))
m_on_cb = CatBoostClassifier()
m_on_cb.load_model(str(ASSETS / "onset.cbm"))

# 2. Load free speed per link
ffs = {}
for p in (DATA / "corridors").iterdir():
    if p.is_dir():
        fd = p / "network" / "fd_parameters.csv"
        if fd.exists():
            df = pd.read_csv(fd, dtype={"link_id": str})
            for _, r in df.iterrows():
                ffs[(p.name, str(r["link_id"]))] = float(r.get("free_speed_kmh", 105.0))

# 3. Load bottleneck priors
priors = {}
for p in PRIORS_DIR.glob("*_bottleneck_prior.json"):
    panel = p.name.replace("_bottleneck_prior.json", "")
    priors[panel] = json.loads(p.read_text(encoding="utf-8"))

def get_slot(ts):
    return int(ts.weekday() * 288 + ts.hour * 12 + ts.minute // 5)

print("Extracting predictions and ground truth for all 40 train onset windows...")
t0 = time.time()
window_records = []

for panel_no, name in enumerate(panel_names()):
    panel_onset = q_onset_module.Panel(name, panel_no)
    prior = priors.get(name, {})
    link_priors = prior.get("links", {})
    link_overall = prior.get("link_overall_freq", {})

    w_file = DATA / "task2" / name / "train" / "window_index.csv"
    if not w_file.exists():
        continue
    windows = pd.read_csv(w_file, dtype={"window_id": str})
    onset = windows[windows.condition == "queue_onset"]
    history = pd.read_parquet(DATA / "task2" / name / "train" / "window_history.parquet")

    # Read unmasked state
    state_files = sorted((DATA / "corridors" / name / "train" / "mainline_states").glob("**/*.parquet"))
    s_dfs = [pd.read_parquet(sf, columns=["timestamp", "link_id", "speed_kmh"]) for sf in state_files]
    state_df = pd.concat(s_dfs, ignore_index=True)
    state_df["timestamp"] = pd.to_datetime(state_df["timestamp"], utc=True)
    state_df["link_id"] = state_df["link_id"].astype(str)

    for row in onset.itertuples(index=False):
        origin = pd.Timestamp(row.forecast_origin)
        times = pd.date_range(end=origin - pd.Timedelta(minutes=5), periods=HISTORY, freq="5min", tz="UTC")
        index = pd.Index(times.strftime("%Y-%m-%dT%H:%M:%SZ"))
        h = history[history.window_id.astype(str) == str(row.window_id)].copy()
        h["link_id"] = h.link_id.astype(str)

        sp = h.pivot(index="timestamp", columns="link_id", values="speed_kmh").reindex(index=index, columns=panel_onset.links).to_numpy(dtype=np.float32)
        fl = h.pivot(index="timestamp", columns="link_id", values="flow_vph").reindex(index=index, columns=panel_onset.links).to_numpy(dtype=np.float32)
        observed = h.pivot(index="timestamp", columns="link_id", values="is_score_eligible").reindex(index=index, columns=panel_onset.links).fillna(0).to_numpy(dtype=bool)

        X = panel_onset.features(sp, fl, observed, origin, "queue_onset")
        p_lgb = m_on_lgb.predict(X, num_threads=6)[-len(panel_onset.links):]
        p_cb = m_on_cb.predict_proba(X, thread_count=6)[:, 1][-len(panel_onset.links):]

        slot_t30 = get_slot(origin + pd.Timedelta(minutes=30))
        eb_priors = np.zeros(len(panel_onset.links), dtype=np.float64)
        for li, lid in enumerate(panel_onset.links):
            str_lid = str(lid)
            eb_priors[li] = link_priors.get(str_lid, {}).get(str(slot_t30), link_overall.get(str_lid, 0.05))

        # Ground truth at T+30
        t30 = origin + pd.Timedelta(minutes=30)
        snap = state_df[state_df.timestamp == t30]
        y_true = np.zeros(len(panel_onset.links), dtype=bool)
        if not snap.empty:
            speed_map = dict(zip(snap["link_id"], snap["speed_kmh"]))
            for li, lid in enumerate(panel_onset.links):
                v_true = speed_map.get(str(lid), np.nan)
                v_cut = 0.60 * ffs.get((name, str(lid)), 105.0)
                if np.isfinite(v_true) and v_true <= v_cut:
                    y_true[li] = True

        window_records.append({
            "panel": name,
            "window_id": str(row.window_id),
            "p_lgb": p_lgb,
            "p_cb": p_cb,
            "eb_priors": eb_priors,
            "y_true": y_true,
            "n_links": len(panel_onset.links),
        })

print(f"Extracted data for {len(window_records)} windows in {time.time()-t0:.1f}s.")

# Grid search weights and thresholds
results = []
for w_lgb in np.linspace(0.0, 1.0, 21):
    w_cb = 1.0 - w_lgb
    for tau in np.arange(0.45, 0.66, 0.02):
        ious = []
        for wr in window_records:
            p_blend = w_lgb * wr["p_lgb"] + w_cb * wr["p_cb"]
            pred = p_blend >= tau
            if not pred.any():
                adj = p_blend * (1.0 + 0.30 * wr["eb_priors"])
                pred[np.argmax(adj)] = True

            y = wr["y_true"]
            inter = np.sum(pred & y)
            union = np.sum(pred | y)
            if union == 0:
                iou = 1.0 if not pred.any() and not y.any() else 0.0
            else:
                iou = inter / union
            ious.append(iou)

        mean_iou = float(np.mean(ious))
        results.append({
            "w_lgb": round(float(w_lgb), 2),
            "w_cb": round(float(w_cb), 2),
            "tau": round(float(tau), 2),
            "mean_iou": mean_iou,
        })

df_res = pd.DataFrame(results).sort_values("mean_iou", ascending=False)
print("\n" + "=" * 65)
print(">>> TOP 20 ONSET BLEND CONFIGURATIONS BY EXACT WINDOW IoU <<<")
print("=" * 65)
print(df_res.head(20).to_string(index=False))

# Show baseline 50/50 @ 0.55
baseline_row = df_res[(df_res.w_lgb == 0.50) & (df_res.tau == 0.55)]
print("\n--- S24 Baseline (50% LGB + 50% CB @ 0.55) ---")
print(baseline_row.to_string(index=False))
