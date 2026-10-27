from pathlib import Path
import pandas as pd
import numpy as np
import sys

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))
from official_toolkit.src.task2.build_task2_persistence_submission import read_window_index, read_queue_template
from official_toolkit.src.task2.queue_utils import thresholds

root = Path("data/kaggle_public")
splits = ["train"]
windows = read_window_index(root, splits)
template = read_queue_template(root, splits)
template["window_id"] = template.window_id.astype(str)
template["link_id"] = template.link_id.astype(str)
template["timestamp"] = pd.to_datetime(template.timestamp, utc=True)

thresholds_by_panel = {}
for panel in sorted(windows.panel.astype(str).unique()):
    panel_dir = root / "corridors" / panel
    links = pd.read_csv(panel_dir / "network" / "links.csv")
    links["link_id"] = links.link_id.astype(str)
    thresholds_by_panel[panel], _ = thresholds(panel_dir, links)

truth_rows = []
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
truth_df.to_parquet("scratch/train_queue_truth.parquet", index=False)
print(f"Saved scratch/train_queue_truth.parquet: {len(truth_df)} rows.")
