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

print(f"Loaded {len(windows)} windows for {splits}")

# Cache thresholds per panel
thresholds_by_panel = {}
for panel in sorted(windows.panel.astype(str).unique()):
    panel_dir = root / "corridors" / panel
    links = pd.read_csv(panel_dir / "network" / "links.csv")
    links["link_id"] = links.link_id.astype(str)
    thresholds_by_panel[panel], _ = thresholds(panel_dir, links)

# Now build ground truth for all train windows
print("Building ground truth for train windows...")
truth_rows = []
for panel in sorted(windows.panel.unique()):
    panel_w = windows[windows.panel == panel]
    mainline_dir = root / "corridors" / panel / "train" / "mainline_states"
    # Load columns link_id, timestamp, speed_kmh
    m_df = pd.read_parquet(mainline_dir, columns=["link_id", "timestamp", "speed_kmh"])
    m_df["link_id"] = m_df["link_id"].astype(str)
    m_df["timestamp"] = pd.to_datetime(m_df["timestamp"], utc=True)
    m_indexed = m_df.set_index(["timestamp", "link_id"])["speed_kmh"]
    
    thresh = thresholds_by_panel[panel]
    
    for _, w in panel_w.iterrows():
        wid = str(w.window_id)
        w_tmpl = template[template.window_id == wid].copy()
        
        # Look up speed_kmh in m_indexed
        # Construct MultiIndex for lookup
        keys = pd.MultiIndex.from_arrays([w_tmpl["timestamp"], w_tmpl["link_id"]])
        w_tmpl["speed_kmh"] = m_indexed.reindex(keys).values
        w_tmpl["v_cut"] = w_tmpl["link_id"].map(thresh).fillna(63.0)
        
        # Ground truth queue: speed <= v_cut & speed.notna()
        w_tmpl["queue_true"] = (w_tmpl["speed_kmh"] <= w_tmpl["v_cut"]) & w_tmpl["speed_kmh"].notna()
        # is_score_eligible: in template, all unmasked cells in target are score eligible if present
        # In fact, let's check if speed is present
        w_tmpl["is_score_eligible"] = w_tmpl["speed_kmh"].notna()
        truth_rows.append(w_tmpl[["window_id", "timestamp", "link_id", "queue_true", "is_score_eligible"]])

truth_df = pd.concat(truth_rows, ignore_index=True)
print(f"Ground truth built: {len(truth_df)} rows. Total queued cells: {truth_df['queue_true'].sum()}")

# Analyze queue prevalence by condition
cond_map = windows.set_index("window_id")["condition"].to_dict()
truth_df["condition"] = truth_df["window_id"].map(cond_map)

print("\n--- Ground Truth Statistics by Condition ---")
for cond, grp in truth_df.groupby("condition"):
    n_windows = grp["window_id"].nunique()
    n_queued = grp["queue_true"].sum()
    n_eligible = grp["is_score_eligible"].sum()
    print(f"Condition '{cond}': {n_windows} windows, {n_queued} queued cells / {n_eligible} eligible cells ({n_queued/n_eligible:.2%})")

# Check distribution of queued cells per window in onset windows
onset_truth = truth_df[truth_df["condition"] == "queue_onset"]
onset_counts = onset_truth.groupby("window_id")["queue_true"].sum()
print("\nOnset windows queued cells distribution (min, 25%, 50%, 75%, max):")
print(onset_counts.describe())
print(f"Onset windows with ZERO queued cells: {(onset_counts == 0).sum()} / {len(onset_counts)}")
