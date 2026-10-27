from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
import sys

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))
from official_toolkit.src.task2.build_task2_persistence_submission import read_window_index, read_queue_template, read_window_history
from official_toolkit.src.task2.queue_utils import thresholds
from official_toolkit.src.task2.score_task2 import score_window
from src.task2.train_and_predict import extract_window_features

root = Path("data/kaggle_public")
splits = ["train"]
windows = read_window_index(root, splits)
template = read_queue_template(root, splits)
history = read_window_history(root, splits)
template["window_id"] = template.window_id.astype(str)
template["link_id"] = template.link_id.astype(str)
template["timestamp"] = pd.to_datetime(template.timestamp, utc=True)
history["window_id"] = history.window_id.astype(str)
history["timestamp"] = pd.to_datetime(history.timestamp, utc=True)
history["link_id"] = history.link_id.astype(str)
history["speed_kmh"] = pd.to_numeric(history.speed_kmh, errors="coerce")

truth_df = pd.read_parquet("scratch/train_queue_truth.parquet")
truth_df["timestamp"] = pd.to_datetime(truth_df["timestamp"], utc=True)

cond_map = windows.set_index("window_id")["condition"].to_dict()
panel_map = windows.set_index("window_id")["panel"].to_dict()
truth_df["condition"] = truth_df["window_id"].map(cond_map)
truth_df["panel"] = truth_df["window_id"].map(panel_map)

# Extract rich window history features per link
print("Extracting rich features for onset modeling...")
records = []
for wid, w_df in history.groupby("window_id", sort=False):
    cond = cond_map.get(wid)
    panel = panel_map.get(wid)
    t_origin = w_df["timestamp"].max()
    for lid, l_df in w_df.groupby("link_id", sort=False):
        l_sorted = l_df.sort_values("timestamp")
        speeds = l_sorted["speed_kmh"].dropna().to_numpy()
        if len(speeds) >= 12:
            s_curr = speeds[-1]
            s_5 = speeds[-2]
            s_10 = speeds[-3]
            s_15 = speeds[-4]
            s_30 = speeds[-7]
            s_60 = speeds[0]
            records.append({
                "window_id": wid,
                "link_id": lid,
                "panel": panel,
                "condition": cond,
                "s_curr": s_curr,
                "s_min": np.min(speeds),
                "s_mean": np.mean(speeds),
                "s_std": np.std(speeds),
                "drop_5": s_5 - s_curr,
                "drop_15": s_15 - s_curr,
                "drop_30": s_30 - s_curr,
                "drop_60": s_60 - s_curr,
            })
feat_df = pd.DataFrame(records)

# Target: did link queue at T+30?
t30_truth = truth_df.copy()
window_origins = history.groupby("window_id")["timestamp"].max().to_dict()
t30_truth["t_origin"] = t30_truth["window_id"].map(window_origins)
t30_truth["delta_min"] = (t30_truth["timestamp"] - t30_truth["t_origin"]).dt.total_seconds() / 60.0
t30_target = t30_truth[t30_truth["delta_min"] >= 30].groupby(["window_id", "link_id"])["queue_true"].max().reset_index()

onset_feat = feat_df[feat_df["condition"] == "queue_onset"].merge(t30_target, on=["window_id", "link_id"], how="left")
onset_feat["queue_true"] = onset_feat["queue_true"].fillna(0).astype(int)
print(f"Onset samples: {len(onset_feat)}, Positives: {onset_feat['queue_true'].sum()}")

# Inspect correlations
num_cols = ["s_curr", "s_min", "s_mean", "s_std", "drop_5", "drop_15", "drop_30", "drop_60"]
print("\nCorrelations with queue_true:")
print(onset_feat[num_cols + ["queue_true"]].corr()["queue_true"])
