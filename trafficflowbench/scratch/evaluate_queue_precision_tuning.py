import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from catboost import CatBoostClassifier

import os
import sys
ROOT = Path("trafficflowbench")
DATA = ROOT / "data" / "kaggle_public"
ASSETS = ROOT / "kaggle_research" / "anubis_assets"
MODELS_DIR = ROOT / "outputs" / "models"

os.environ["TFB_DATA_ROOT"] = str(DATA)
os.environ["TFB_OUTPUT_ROOT"] = str(ROOT / "scratch")
os.environ["TFB_THREADS"] = "6"

sys.path.insert(0, str(ASSETS))
import queue_ongoing as q_ongoing_module
from queue_features import panel_names, HISTORY, HORIZON

gt = pd.read_parquet(ROOT / "outputs" / "ground_truth" / "task2_train_ground_truth.parquet")
ong_gt = gt[gt["condition"] == "queue_ongoing"].copy()

print("Loading models...")
m_ong_lgb = lgb.Booster(model_file=str(ASSETS / "ongoing_lgb.txt"))
m_ong_cb = CatBoostClassifier()
m_ong_cb.load_model(str(MODELS_DIR / "ongoing_cb.cbm"))

# Compute predictions for all 40 train ongoing windows
panels = panel_names()
records = []

for no, name in enumerate(panels):
    panel = q_ongoing_module.Panel(name, no)
    folder = DATA / "task2" / name / "train"
    if not (folder / "window_index.csv").exists():
        continue
    windows = pd.read_csv(folder / "window_index.csv", dtype={"window_id": str})
    hist = pd.read_parquet(folder / "window_history.parquet")
    ongoing_wins = windows[windows.condition == "queue_ongoing"]
    
    for w in ongoing_wins.itertuples(index=False):
        origin = pd.Timestamp(w.forecast_origin)
        times = pd.date_range(end=origin - pd.Timedelta(minutes=5), periods=HISTORY, freq="5min", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        target_times = [origin + pd.Timedelta(minutes=5 * h) for h in range(1, HORIZON + 1)]
        h_df = hist[hist.window_id == w.window_id].copy()
        h_df["link_id"] = h_df.link_id.astype(str)
        
        mat = lambda c: h_df.pivot(index="timestamp", columns="link_id", values=c).reindex(index=times, columns=panel.links).to_numpy(dtype=np.float32)
        speed, flow, elig = mat("speed_kmh"), mat("flow_vph"), mat("is_score_eligible")
        
        X = panel.features(speed, flow, np.nan_to_num(elig, nan=0).astype(bool), origin, w.condition)
        p_lgb = m_ong_lgb.predict(X, num_threads=6)
        p_cb = m_ong_cb.predict_proba(X, thread_count=6)[:, 1]
        p_s24 = 0.88 * p_lgb + 0.12 * p_cb
        
        for h_idx in range(HORIZON):
            for m_idx in range(len(panel.links)):
                records.append({
                    "window_id": str(w.window_id),
                    "timestamp": target_times[h_idx],
                    "link_id": str(panel.links[m_idx]),
                    "panel": name,
                    "condition": "queue_ongoing",
                    "h": h_idx + 1,
                    "prob": float(p_s24[h_idx * len(panel.links) + m_idx]),
                })

pred_df = pd.DataFrame(records)
pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"], utc=True)
print(f"Generated predictions for {len(pred_df):,} train ongoing cells.")

# Merge with ground truth
merged = ong_gt[ong_gt["is_score_eligible"].astype(bool)].merge(
    pred_df[["window_id", "timestamp", "link_id", "prob", "h"]],
    on=["window_id", "timestamp", "link_id"],
    how="inner"
)

print(f"Scorable cells in ground truth: {len(merged):,}")

# Grid search threshold tau
print("\n" + "=" * 65)
print(f"{'Threshold (tau)':<18} | {'Overall Ongoing IoU':<22} | {'Positives':<10}")
print("-" * 65)

best_tau = 0.45
best_iou = 0.0

for tau in np.arange(0.35, 0.551, 0.025):
    merged["pred"] = (merged["prob"] >= tau).astype(int)
    window_ious = []
    for wid, grp in merged.groupby("window_id"):
        true = grp["queue_true"].to_numpy().astype(bool)
        pred = grp["pred"].to_numpy().astype(bool)
        inter = (true & pred).sum()
        union = (true | pred).sum()
        window_ious.append(1.0 if union == 0 else inter / union)
    iou = float(np.mean(window_ious))
    pos = int(merged["pred"].sum())
    marker = " *** BEST ***" if iou > best_iou else ""
    if iou > best_iou:
        best_iou = iou
        best_tau = tau
    print(f"{tau:<18.3f} | {iou:<22.5f} | {pos:<10}{marker}")

print("=" * 65)
print(f"Optimal Threshold: tau = {best_tau:.3f} with IoU = {best_iou:.5f}")

