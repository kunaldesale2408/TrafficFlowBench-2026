"""Deep Diagnostic Analysis of the 1,146 Differing Ongoing Queue Cells:
Compare 0.87141 benchmark ongoing queue vs S24 champion ongoing queue.
"""
from __future__ import annotations

import glob
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("trafficflowbench").resolve()
DATA = ROOT / "data" / "kaggle_public"
SUB_87 = ROOT / "kaggle_research" / "semwal_087" / "output" / "submission.csv"
SUB_S24 = ROOT / "outputs" / "submissions" / "submission_s24.csv"
KEY_FILE = DATA / "submission_key.csv"

print("Loading submission key...")
key = pd.read_csv(KEY_FILE, low_memory=False)
qu_mask = key["task"] == "queue"
key_qu = key[qu_mask].copy()

# Map window conditions
wi_files = list((DATA / "task2").glob("*/*/window_index.csv"))
wi_df = pd.concat([pd.read_csv(f) for f in wi_files], ignore_index=True)
cond_map = dict(zip(wi_df["window_id"].astype(str), wi_df["condition"].astype(str)))
origin_map = dict(zip(wi_df["window_id"].astype(str), wi_df["forecast_origin"].astype(str)))

key_qu["condition"] = key_qu["window_id"].astype(str).map(cond_map)
key_qu["forecast_origin"] = key_qu["window_id"].astype(str).map(origin_map)
key_qu["h"] = ((pd.to_datetime(key_qu["timestamp"], utc=True) - pd.to_datetime(key_qu["forecast_origin"], utc=True)).dt.total_seconds() // 300).astype(int)

print("Loading submissions...")
sub_87 = pd.read_csv(SUB_87)
sub_s24 = pd.read_csv(SUB_S24)

q_87 = sub_87.loc[qu_mask, "queue_pred"].to_numpy().astype(int)
q_s24 = sub_s24.loc[qu_mask, "queue_pred"].to_numpy().astype(int)

key_qu["pred_87"] = q_87
key_qu["pred_s24"] = q_s24

is_ong = key_qu["condition"] == "queue_ongoing"
df_ong = key_qu[is_ong].copy()

diff_mask = df_ong["pred_87"] != df_ong["pred_s24"]
df_diff = df_ong[diff_mask].copy()

print(f"\nTotal ongoing cells: {len(df_ong):,}")
print(f"Total differing ongoing cells: {len(df_diff):,}")

# Breakdown of differing types
# Type A: 87 says 1, S24 says 0
type_a = (df_diff["pred_87"] == 1) & (df_diff["pred_s24"] == 0)
# Type B: 87 says 0, S24 says 1
type_b = (df_diff["pred_87"] == 0) & (df_diff["pred_s24"] == 1)

print(f"Type A (0.87141=1, S24=0): {type_a.sum():,} cells ({type_a.sum()/len(df_diff)*100:.1f}%)")
print(f"Type B (0.87141=0, S24=1): {type_b.sum():,} cells ({type_b.sum()/len(df_diff)*100:.1f}%)")

# Horizon distribution of differences
print("\n" + "=" * 60)
print(">>> DIFFERENCES BY FORECAST HORIZON (h = 1..12) <<<")
print("=" * 60)
h_dist = df_diff.groupby("h").agg(
    total_diff=("pred_87", "count"),
    type_A_87pos=("pred_87", lambda x: (x == 1).sum()),
    type_B_s24pos=("pred_s24", lambda x: (x == 1).sum())
)
print(h_dist)

# Corridor/Panel distribution of differences
print("\n" + "=" * 60)
print(">>> DIFFERENCES BY PANEL <<<")
print("=" * 60)
p_dist = df_diff.groupby("panel").agg(
    total_diff=("pred_87", "count"),
    type_A_87pos=("pred_87", lambda x: (x == 1).sum()),
    type_B_s24pos=("pred_s24", lambda x: (x == 1).sum())
)
print(p_dist)

# Split distribution (validation vs private)
df_diff["split"] = np.where(df_diff["timestamp"].str[:7] == "2031-03", "validation", "private")
print("\n" + "=" * 60)
print(">>> DIFFERENCES BY SPLIT <<<")
print("=" * 60)
s_dist = df_diff.groupby("split").agg(
    total_diff=("pred_87", "count"),
    type_A_87pos=("pred_87", lambda x: (x == 1).sum()),
    type_B_s24pos=("pred_s24", lambda x: (x == 1).sum())
)
print(s_dist)
