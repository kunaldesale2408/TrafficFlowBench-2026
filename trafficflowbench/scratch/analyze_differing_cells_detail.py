"""Examine the 1,146 differing cells in detail:
What are the underlying physical speeds in S24 and Semwal for these cells?
Are they clearly congested (v < 35 km/h) or borderline (45-65 km/h)?
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

key = pd.read_csv(KEY_FILE, low_memory=False)
qu_mask = key["task"] == "queue"
key_qu = key[qu_mask].copy()

wi_files = list((DATA / "task2").glob("*/*/window_index.csv"))
wi_df = pd.concat([pd.read_csv(f) for f in wi_files], ignore_index=True)
cond_map = dict(zip(wi_df["window_id"].astype(str), wi_df["condition"].astype(str)))
origin_map = dict(zip(wi_df["window_id"].astype(str), wi_df["forecast_origin"].astype(str)))

key_qu["condition"] = key_qu["window_id"].astype(str).map(cond_map)
key_qu["forecast_origin"] = key_qu["window_id"].astype(str).map(origin_map)
key_qu["h"] = ((pd.to_datetime(key_qu["timestamp"], utc=True) - pd.to_datetime(key_qu["forecast_origin"], utc=True)).dt.total_seconds() // 300).astype(int)

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

# Load speed cutoffs for each link to see the official congestion threshold
v_cuts = {}
for p in df_diff["panel"].unique():
    pdir = DATA / "corridors" / p / "network"
    fd = pd.read_csv(pdir / "fd_parameters.csv")
    fd["link_id"] = fd["link_id"].astype(str)
    # v_cut in official queue is 0.55 * vf or similar
    if "free_speed_kmh" in fd.columns:
        m = fd.groupby("link_id")["free_speed_kmh"].mean()
        for lid, vf in m.items():
            v_cuts[lid] = 0.55 * vf

df_diff["v_cut_approx"] = df_diff["link_id"].map(v_cuts).fillna(60.0)

print("=" * 70)
print(f">>> DETAILED ANALYSIS OF {len(df_diff):,} DIFFERING ONGOING CELLS <<<")
print("=" * 70)

# What are the most frequent links in df_diff?
print("\nTop 10 Links with most disagreements:")
link_counts = df_diff.groupby(["panel", "link_id"]).agg(
    disagreements=("pred_87", "count"),
    pred_87_pos=("pred_87", lambda x: (x == 1).sum()),
    pred_s24_pos=("pred_s24", lambda x: (x == 1).sum()),
    v_cut=("v_cut_approx", "first")
).sort_values("disagreements", ascending=False)

print(link_counts.head(15))

# What are the most frequent windows?
print("\nTop 10 Windows with most disagreements:")
win_counts = df_diff.groupby(["panel", "window_id"]).agg(
    disagreements=("pred_87", "count"),
    pred_87_pos=("pred_87", lambda x: (x == 1).sum()),
    pred_s24_pos=("pred_s24", lambda x: (x == 1).sum())
).sort_values("disagreements", ascending=False)

print(win_counts.head(10))
