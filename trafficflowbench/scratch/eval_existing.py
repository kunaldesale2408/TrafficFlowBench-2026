from pathlib import Path
import pandas as pd
import numpy as np
import sys

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))
from official_toolkit.src.task2.build_task2_persistence_submission import read_window_index, read_queue_template, read_window_history
from official_toolkit.src.task2.queue_utils import thresholds
from official_toolkit.src.task2.score_task2 import score_window
from src.task2.train_and_predict import extract_window_features, predict_queues

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

# Load ground truth
truth_rows = []
thresholds_by_panel = {}
for panel in sorted(windows.panel.astype(str).unique()):
    panel_dir = root / "corridors" / panel
    links = pd.read_csv(panel_dir / "network" / "links.csv")
    links["link_id"] = links.link_id.astype(str)
    thresholds_by_panel[panel], _ = thresholds(panel_dir, links)

for panel in sorted(windows.panel.unique()):
    panel_w = windows[windows.panel == panel]
    mainline_dir = root / "corridors" / panel / "train" / "mainline_states"
    m_df = pd.read_parquet(mainline_dir, columns=["link_id", "timestamp", "speed_kmh"])
    m_df["link_id"] = m_df["link_id"].astype(str)
    m_df["timestamp"] = pd.to_datetime(m_df["timestamp"], utc=True)
    m_indexed = m_df.set_index(["timestamp", "link_id"])["speed_kmh"]
    thresh = thresholds_by_panel[panel]
    
    for _, w in panel_w.iterrows():
        wid = str(w.window_id)
        w_tmpl = template[template.window_id == wid].copy()
        keys = pd.MultiIndex.from_arrays([w_tmpl["timestamp"], w_tmpl["link_id"]])
        w_tmpl["speed_kmh"] = m_indexed.reindex(keys).values
        w_tmpl["v_cut"] = w_tmpl["link_id"].map(thresh).fillna(63.0)
        w_tmpl["queue_true"] = (w_tmpl["speed_kmh"] <= w_tmpl["v_cut"]) & w_tmpl["speed_kmh"].notna()
        w_tmpl["is_score_eligible"] = w_tmpl["speed_kmh"].notna()
        truth_rows.append(w_tmpl[["window_id", "timestamp", "link_id", "queue_true", "is_score_eligible"]])

truth_df = pd.concat(truth_rows, ignore_index=True)

# Run existing predict_queues
panel_by_window = windows.set_index("window_id")["panel"].astype(str).to_dict()
cond_col = "condition" if "condition" in windows.columns else ("window_type" if "window_type" in windows.columns else None)
type_by_window = windows.set_index("window_id")[cond_col].astype(str).to_dict() if cond_col else {}
window_origins = history.groupby("window_id")["timestamp"].max().to_dict()

features_df = extract_window_features(history, thresholds_by_panel, panel_by_window, type_by_window)
preds = predict_queues(template, features_df, window_origins)

# Score
rows = []
for w in windows.itertuples(index=False):
    t = truth_df[truth_df.window_id == w.window_id]
    s = preds[preds.window_id == w.window_id]
    iou, missing, invalid = score_window(t, s)
    rows.append({"window_id": w.window_id, "panel": w.panel, "family_id": w.family_id, "condition": w.condition, "IoU_ST": iou})
res = pd.DataFrame(rows)

print("\n=== Evaluation: Existing Code in src/task2 ===")
for cond, g in res.groupby("condition"):
    print(f"Condition {cond}: IoU = {g['IoU_ST'].mean():.4f}")

pc = res.groupby(["panel", "condition"])["IoU_ST"].mean().reset_index()
pm = pc.groupby("panel")["IoU_ST"].mean().reset_index()
overall = pm["IoU_ST"].mean()
print(f"Overall S_queue: {overall:.4f}")
