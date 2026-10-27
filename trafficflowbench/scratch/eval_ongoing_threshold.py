"""Evaluate ongoing queue thresholds across [0.44, 0.46, 0.47, 0.48, 0.49, 0.50, 0.52] on validation windows.
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
from catboost import CatBoostClassifier

ROOT = Path("trafficflowbench").resolve()
DATA = ROOT / "data" / "kaggle_public"
ASSETS = ROOT / "kaggle_research" / "anubis_assets"
MODELS_DIR = ROOT / "outputs" / "models"

os.environ["TFB_DATA_ROOT"] = str(DATA)
os.environ["TFB_OUTPUT_ROOT"] = str(ROOT / "scratch")
os.environ["TFB_THREADS"] = "6"

sys.path.insert(0, str(ASSETS))
import queue_ongoing as q_ongoing_module
from queue_features import panel_names, HISTORY, HORIZON

def main():
    print("Loading ongoing models...")
    m_ong_lgb = lgb.Booster(model_file=str(ASSETS / "ongoing_lgb.txt"))
    m_ong_cb = CatBoostClassifier()
    m_ong_cb.load_model(str(MODELS_DIR / "ongoing_cb.cbm"))
    
    w_ong_lgb = 0.88
    w_ong_cb = 0.12
    
    panels = panel_names()
    val_probs = []
    val_truths = []
    
    for panel_no, name in enumerate(panels):
        panel_ongoing = q_ongoing_module.Panel(name, panel_no)
        folder = DATA / "task2" / name / "train"
        windows = pd.read_csv(folder / "window_index.csv", dtype={"window_id": str})
        history = pd.read_parquet(folder / "window_history.parquet")
        truth = pd.read_parquet(folder / "window_truth.parquet")
        
        ongoing_windows = windows[windows.condition == "queue_ongoing"]
        
        for row in ongoing_windows.itertuples(index=False):
            origin = pd.Timestamp(row.forecast_origin)
            times = pd.date_range(end=origin - pd.Timedelta(minutes=5), periods=HISTORY, freq="5min", tz="UTC")
            index = pd.Index(times.strftime("%Y-%m-%dT%H:%M:%SZ"))
            h = history[history.window_id.astype(str) == str(row.window_id)].copy()
            h["link_id"] = h.link_id.astype(str)
            
            sp = h.pivot(index="timestamp", columns="link_id", values="speed_kmh").reindex(index=index, columns=panel_ongoing.links).to_numpy(dtype=np.float32)
            fl = h.pivot(index="timestamp", columns="link_id", values="flow_vph").reindex(index=index, columns=panel_ongoing.links).to_numpy(dtype=np.float32)
            observed = h.pivot(index="timestamp", columns="link_id", values="is_score_eligible").reindex(index=index, columns=panel_ongoing.links).fillna(0).to_numpy(dtype=bool)
            
            X = panel_ongoing.features(sp, fl, observed, origin, "queue_ongoing")
            p_lgb = m_ong_lgb.predict(X, num_threads=6)
            p_cb = m_ong_cb.predict_proba(X, thread_count=6)[:, 1]
            probs = w_ong_lgb * p_lgb + w_ong_cb * p_cb
            
            tr = truth[truth.window_id.astype(str) == str(row.window_id)]
            t_times = [(origin + pd.Timedelta(minutes=5 * s)).strftime("%Y-%m-%dT%H:%M:%SZ") for s in range(1, HORIZON + 1)]
            t_pivot = tr.pivot(index="timestamp", columns="link_id", values="is_queued").reindex(index=t_times, columns=panel_ongoing.links).to_numpy(dtype=bool)
            e_pivot = tr.pivot(index="timestamp", columns="link_id", values="is_score_eligible").reindex(index=t_times, columns=panel_ongoing.links).to_numpy(dtype=bool)
            
            val_probs.append(probs[e_pivot.ravel()])
            val_truths.append(t_pivot.ravel()[e_pivot.ravel()])
            
    all_p = np.concatenate(val_probs)
    all_t = np.concatenate(val_truths)
    print(f"Total evaluated cells across validation ongoing windows: {len(all_p):,}")
    print(f"True positive rate: {np.mean(all_t):.3%}")
    
    print("\n--- Ongoing Queue IoU vs Decision Threshold ---")
    print(f"{'Threshold':<12} | {'IoU':<12} | {'Precision':<12} | {'Recall':<12}")
    print("-" * 55)
    best_iou, best_th = 0.0, 0.0
    for th in [0.42, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.52, 0.55]:
        pred = all_p >= th
        inter = np.sum(pred & all_t)
        union = np.sum(pred | all_t)
        iou = inter / union if union > 0 else 0.0
        prec = inter / np.sum(pred) if np.sum(pred) > 0 else 0.0
        rec = inter / np.sum(all_t) if np.sum(all_t) > 0 else 0.0
        marker = " *** BEST ***" if iou > best_iou else ""
        if iou > best_iou:
            best_iou = iou
            best_th = th
        print(f"{th:<12.2f} | {iou:<12.5f} | {prec:<12.4f} | {rec:<12.4f}{marker}")

if __name__ == "__main__":
    main()
