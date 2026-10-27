from pathlib import Path
import pandas as pd
import numpy as np
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

# Load cached truth
truth_df = pd.read_parquet("scratch/train_queue_truth.parquet")
truth_df["timestamp"] = pd.to_datetime(truth_df["timestamp"], utc=True)

cond_map = windows.set_index("window_id")["condition"].to_dict()
panel_map = windows.set_index("window_id")["panel"].to_dict()
truth_df["condition"] = truth_df["window_id"].map(cond_map)
truth_df["panel"] = truth_df["window_id"].map(panel_map)

# Load thresholds
thresholds_by_panel = {}
for panel in sorted(windows.panel.astype(str).unique()):
    panel_dir = root / "corridors" / panel
    links = pd.read_csv(panel_dir / "network" / "links.csv")
    links["link_id"] = links.link_id.astype(str)
    thresholds_by_panel[panel], _ = thresholds(panel_dir, links)

# Extract features
features_df = extract_window_features(history, thresholds_by_panel, panel_map, cond_map)

# Let's inspect the true queue rows in onset windows
onset_truth = truth_df[(truth_df["condition"] == "queue_onset") & (truth_df["queue_true"] == 1)].copy()

window_origins = history.groupby("window_id")["timestamp"].max().to_dict()
onset_truth["t_origin"] = onset_truth["window_id"].map(window_origins)
onset_truth["delta_min"] = (onset_truth["timestamp"] - onset_truth["t_origin"]).dt.total_seconds() / 60.0

print(f"Total queued cells in onset windows: {len(onset_truth)}")
print("\nBreakdown of true queue cells by delta_min:")
print(onset_truth["delta_min"].value_counts().sort_index())

print("\nBreakdown of true queue cells by panel:")
print(onset_truth["panel"].value_counts())

print("\nTop links that queue in onset windows:")
print(onset_truth.groupby(["panel", "link_id"]).size().sort_values(ascending=False).head(15))

# Merge features with onset_truth to see historical signals
onset_merged = onset_truth.merge(features_df, on=["window_id", "link_id"], how="left")
print("\nSpeed ratios for true queue cells at T (v_curr / v_cut):")
ratio = onset_merged["v_curr"] / onset_merged["v_cut"]
print(ratio.describe())

print("\nDeceleration for true queue cells (decel_15):")
print(onset_merged["decel_15"].describe())
