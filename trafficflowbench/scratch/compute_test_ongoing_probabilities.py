"""Compute Continuous Ongoing Queue Probabilities for Test Split (Validation + Private):
1. p_pub: Anubis public LightGBM ongoing booster
2. p_s24: S24 Champion Dual (88% LGB + 12% CatBoost)
Saves:
scratch/test_ongoing_probabilities.parquet
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier

ROOT = Path("trafficflowbench").resolve()
DATA = ROOT / "data" / "kaggle_public"
ASSETS = ROOT / "kaggle_research" / "anubis_assets"
MODELS_DIR = ROOT / "outputs" / "models"
OUT_PATH = ROOT / "scratch" / "test_ongoing_probabilities.parquet"

os.environ["TFB_DATA_ROOT"] = str(DATA)
os.environ["TFB_OUTPUT_ROOT"] = str(ROOT / "scratch")
os.environ["TFB_THREADS"] = "6"

sys.path.insert(0, str(ASSETS))
import queue_ongoing as q_ongoing_module
from queue_features import panel_names, HISTORY, HORIZON

print("Loading ongoing models...")
m_ong_lgb = lgb.Booster(model_file=str(ASSETS / "ongoing_lgb.txt"))
m_ong_cb = CatBoostClassifier()
m_ong_cb.load_model(str(MODELS_DIR / "ongoing_cb.cbm"))

print("Generating features and predictions across test splits...")
t0 = time.time()
rows = []

for split in ("validation", "private"):
    for no, name in enumerate(panel_names()):
        panel = q_ongoing_module.Panel(name, no)
        folder = DATA / "task2" / name / split
        windows = pd.read_csv(folder / "window_index.csv", dtype={"window_id": str})
        hist = pd.read_parquet(folder / "window_history.parquet")
        
        ongoing_wins = windows[windows.condition == "queue_ongoing"]
        
        for w in ongoing_wins.itertuples(index=False):
            origin = pd.Timestamp(w.forecast_origin)
            times = pd.date_range(end=origin - pd.Timedelta(minutes=5), periods=HISTORY, freq="5min", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
            h = hist[hist.window_id == w.window_id].copy()
            h["link_id"] = h.link_id.astype(str)
            
            mat = lambda c: h.pivot(index="timestamp", columns="link_id", values=c).reindex(index=times, columns=panel.links).to_numpy(dtype=np.float32)
            speed, flow, elig = mat("speed_kmh"), mat("flow_vph"), mat("is_score_eligible")
            
            X = panel.features(speed, flow, np.nan_to_num(elig, nan=0).astype(bool), origin, w.condition)
            
            p_lgb = m_ong_lgb.predict(X, num_threads=6)
            p_cb = m_ong_cb.predict_proba(X, thread_count=6)[:, 1]
            p_s24 = 0.88 * p_lgb + 0.12 * p_cb
            
            rows.append(pd.DataFrame({
                "split": split,
                "panel": name,
                "window_id": w.window_id,
                "h": np.repeat(np.arange(1, HORIZON + 1), len(panel.links)),
                "link_id": np.tile(panel.links, HORIZON),
                "p_lgb": p_lgb,
                "p_cb": p_cb,
                "p_s24": p_s24,
            }))

df_prob = pd.concat(rows, ignore_index=True)
print(f"Computed probabilities for {len(df_prob):,} cells in {time.time()-t0:.1f}s")
df_prob.to_parquet(OUT_PATH, index=False)
print(f"Saved {OUT_PATH}")
