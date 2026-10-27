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

features_df = extract_window_features(history, thresholds_by_panel, panel_map, cond_map)
window_origins = history.groupby("window_id")["timestamp"].max().to_dict()

base_df = template.copy()
feat_indexed = features_df.set_index(["window_id", "link_id"])
base_df = base_df.join(feat_indexed, on=["window_id", "link_id"], how="left")
base_df["t_origin"] = base_df["window_id"].map(window_origins)
base_df["delta_min"] = (base_df["timestamp"] - base_df["t_origin"]).dt.total_seconds() / 60.0

def eval_ongoing(pred_df):
    rows = []
    for w in windows[windows.condition == "queue_ongoing"].itertuples(index=False):
        t = truth_df[truth_df.window_id == w.window_id]
        s = pred_df[pred_df.window_id == w.window_id]
        iou, _, _ = score_window(t, s)
        rows.append({"window_id": w.window_id, "panel": w.panel, "IoU_ST": iou})
    res = pd.DataFrame(rows)
    return res.groupby("panel")["IoU_ST"].mean().mean()

ongoing_mask = base_df["window_type"].astype(str).str.contains("ongoing")

# Grid search on decel threshold and speed ratio threshold
best_score = 0
best_params = None

print("Grid search for ongoing queue forecasting...")
for decel_thresh in [10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 999.0]:
    for ratio_thresh in [1.0, 1.05, 1.10, 1.15, 999.0]:
        for min_delta in [10, 15, 20]:
            df = base_df[["window_id", "timestamp", "link_id"]].copy()
            df["queue_pred"] = 0
            
            q_persist = ongoing_mask & (base_df["is_queued_now"] == 1)
            q_spread = (
                ongoing_mask 
                & (base_df["is_queued_now"] == 0) 
                & (base_df["delta_min"] >= min_delta)
                & ((base_df["decel_15"] >= decel_thresh) | (base_df["v_curr"] <= base_df["v_cut"] * ratio_thresh))
            )
            df.loc[q_persist | q_spread, "queue_pred"] = 1
            score = eval_ongoing(df)
            if score > best_score:
                best_score = score
                best_params = (decel_thresh, ratio_thresh, min_delta)
                print(f"New best: {best_score:.4f} with decel>={decel_thresh}, ratio<={ratio_thresh}, delta>={min_delta}")

print(f"\nOptimal ongoing parameters: decel={best_params[0]}, ratio={best_params[1]}, min_delta={best_params[2]} -> Score: {best_score:.4f}")
