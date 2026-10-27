from pathlib import Path
import pandas as pd
import numpy as np
import sys

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))
from official_toolkit.src.task2.build_task2_persistence_submission import read_window_index, read_queue_template, read_window_history
from official_toolkit.src.task2.queue_utils import thresholds
from official_toolkit.src.task2.score_task2 import score_window

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

# Load ground truth from previous step
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
        # Eligible in official scoring: speed_kmh is notna
        w_tmpl["is_score_eligible"] = w_tmpl["speed_kmh"].notna()
        truth_rows.append(w_tmpl[["window_id", "timestamp", "link_id", "queue_true", "is_score_eligible"]])

truth_df = pd.concat(truth_rows, ignore_index=True)

# Build Official Persistence predictions
panel_by_window = windows.set_index("window_id").panel.astype(str).to_dict()
targets = []
for window_id, h in history.groupby("window_id", sort=True):
    panel = panel_by_window.get(str(window_id))
    if panel is None:
        continue
    last_time = h.timestamp.max()
    at = h[h.timestamp.eq(last_time)].copy()
    threshold = thresholds_by_panel[panel]
    at["queue_now"] = at.speed_kmh <= at.link_id.map(threshold).fillna(63.0)
    at["queue_now"] &= at.is_score_eligible.astype(bool)
    q = at.groupby("link_id").queue_now.any().to_dict()
    target = template[template.window_id == str(window_id)][["window_id", "timestamp", "link_id"]].copy()
    target["queue_pred"] = target.link_id.map(q).fillna(False).astype(int)
    targets.append(target)

persistence_sub = pd.concat(targets, ignore_index=True)

# Score persistence sub against truth
def eval_sub(sub_df, name="Submission"):
    rows = []
    for w in windows.itertuples(index=False):
        t = truth_df[truth_df.window_id == w.window_id]
        s = sub_df[sub_df.window_id == w.window_id]
        iou, missing, invalid = score_window(t, s)
        rows.append({"window_id": w.window_id, "panel": w.panel, "family_id": w.family_id, "condition": w.condition, "IoU_ST": iou})
    res = pd.DataFrame(rows)
    
    print(f"\n=== Evaluation: {name} ===")
    for cond, g in res.groupby("condition"):
        print(f"Condition {cond}: IoU = {g['IoU_ST'].mean():.4f}")
    
    pc = res.groupby(["panel", "condition"])["IoU_ST"].mean().reset_index()
    pm = pc.groupby("panel")["IoU_ST"].mean().reset_index()
    overall = pm["IoU_ST"].mean()
    print(f"Overall S_queue: {overall:.4f}")
    return res

eval_sub(persistence_sub, "Official Persistence Baseline")
