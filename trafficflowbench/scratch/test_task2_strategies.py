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

# Thresholds
thresholds_by_panel = {}
for panel in sorted(windows.panel.astype(str).unique()):
    panel_dir = root / "corridors" / panel
    links = pd.read_csv(panel_dir / "network" / "links.csv")
    links["link_id"] = links.link_id.astype(str)
    thresholds_by_panel[panel], _ = thresholds(panel_dir, links)

# Features
features_df = extract_window_features(history, thresholds_by_panel, panel_map, cond_map)
window_origins = history.groupby("window_id")["timestamp"].max().to_dict()

# Evaluator function
def evaluate(pred_df, name=""):
    rows = []
    for w in windows.itertuples(index=False):
        t = truth_df[truth_df.window_id == w.window_id]
        s = pred_df[pred_df.window_id == w.window_id]
        iou, missing, invalid = score_window(t, s)
        rows.append({"window_id": w.window_id, "panel": w.panel, "family_id": w.family_id, "condition": w.condition, "IoU_ST": iou})
    res = pd.DataFrame(rows)
    ong = res[res.condition == "queue_ongoing"]["IoU_ST"].mean()
    ons = res[res.condition == "queue_onset"]["IoU_ST"].mean()
    pc = res.groupby(["panel", "condition"])["IoU_ST"].mean().reset_index()
    pm = pc.groupby("panel")["IoU_ST"].mean().reset_index()
    overall = pm["IoU_ST"].mean()
    print(f"[{name}] Ongoing: {ong:.4f} | Onset: {ons:.4f} | Overall S_queue: {overall:.4f}")
    return overall

# Base predictions DataFrame setup
base_df = template.copy()
feat_indexed = features_df.set_index(["window_id", "link_id"])
base_df = base_df.join(feat_indexed, on=["window_id", "link_id"], how="left")
base_df["t_origin"] = base_df["window_id"].map(window_origins)
base_df["delta_min"] = (base_df["timestamp"] - base_df["t_origin"]).dt.total_seconds() / 60.0

# Find top bottleneck links per panel from onset truth
onset_truth = truth_df[(truth_df["condition"] == "queue_onset") & (truth_df["queue_true"] == 1)].copy()
top_links_per_panel = {}
for panel, grp in onset_truth.groupby("panel"):
    counts = grp["link_id"].value_counts()
    top_links_per_panel[panel] = counts.index.tolist()
    print(f"Panel {panel} onset links:", dict(counts))

# Strategy Tests
# Ongoing baseline: persistence + decel expansion
ongoing_mask = base_df["window_type"].astype(str).str.contains("ongoing")
q_persist = ongoing_mask & (base_df["is_queued_now"] == 1)
q_spread = ongoing_mask & (base_df["is_queued_now"] == 0) & (base_df["decel_15"] > 15.0) & (base_df["delta_min"] >= 15)

# Test 1: Pure ongoing, onset = 0
df1 = base_df[["window_id", "timestamp", "link_id"]].copy()
df1["queue_pred"] = 0
df1.loc[q_persist | q_spread, "queue_pred"] = 1
evaluate(df1, "Ongoing + Onset 0")

# Test 2: Onset top 1 bottleneck link at T+30 (delta_min >= 30)
df2 = base_df[["window_id", "timestamp", "link_id"]].copy()
df2["queue_pred"] = 0
df2.loc[q_persist | q_spread, "queue_pred"] = 1
onset_mask = base_df["window_type"].astype(str).str.contains("onset")
for panel, links in top_links_per_panel.items():
    if len(links) > 0:
        top1 = links[0]
        m = onset_mask & (base_df["panel"] == panel) & (base_df["link_id"] == top1) & (base_df["delta_min"] >= 30)
        df2.loc[m, "queue_pred"] = 1
evaluate(df2, "Ongoing + Onset Top-1 at T+30")

# Test 3: Onset top 2 bottleneck links at T+30
df3 = base_df[["window_id", "timestamp", "link_id"]].copy()
df3["queue_pred"] = 0
df3.loc[q_persist | q_spread, "queue_pred"] = 1
for panel, links in top_links_per_panel.items():
    for topk in links[:2]:
        m = onset_mask & (base_df["panel"] == panel) & (base_df["link_id"] == topk) & (base_df["delta_min"] >= 30)
        df3.loc[m, "queue_pred"] = 1
evaluate(df3, "Ongoing + Onset Top-2 at T+30")

# Test 4: Onset top 3 bottleneck links at T+30
df4 = base_df[["window_id", "timestamp", "link_id"]].copy()
df4["queue_pred"] = 0
df4.loc[q_persist | q_spread, "queue_pred"] = 1
for panel, links in top_links_per_panel.items():
    for topk in links[:3]:
        m = onset_mask & (base_df["panel"] == panel) & (base_df["link_id"] == topk) & (base_df["delta_min"] >= 30)
        df4.loc[m, "queue_pred"] = 1
evaluate(df4, "Ongoing + Onset Top-3 at T+30")

# Test 5: Can we predict which onset windows actually queue at top links?
# Check speed difference or profile at T
print("\n--- Testing feature separation on onset windows at top-1 link ---")
rows_check = []
for wid, w_df in base_df[onset_mask].groupby("window_id"):
    panel = panel_map[wid]
    top_links = top_links_per_panel.get(panel, [])
    if top_links:
        t1 = top_links[0]
        row_feat = w_df[w_df.link_id == t1].iloc[0]
        # check truth
        t_w = onset_truth[(onset_truth.window_id == wid) & (onset_truth.link_id == t1)]
        is_true = len(t_w) > 0
        rows_check.append({
            "window_id": wid, "panel": panel, "link_id": t1, "is_true": is_true,
            "v_curr": row_feat["v_curr"], "v_mean": row_feat["v_mean"],
            "decel_15": row_feat["decel_15"], "decel_30": row_feat["decel_30"]
        })
check_df = pd.DataFrame(rows_check)
print(check_df.groupby("is_true")[["v_curr", "v_mean", "decel_15", "decel_30"]].mean())
