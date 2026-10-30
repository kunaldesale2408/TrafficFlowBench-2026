"""Evaluate Consensus Queue Strategy:
Combines 0.87141 benchmark queue with physical shockwave verification from S24.
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

# Let's inspect windows where S24=1 and 87=0
s24_only = (key_qu["condition"] == "queue_ongoing") & (key_qu["pred_87"] == 0) & (key_qu["pred_s24"] == 1)
print(f"Total S24-only ongoing cells: {s24_only.sum():,}")

df_s24_only = key_qu[s24_only]
print("\nS24-only cells by panel:")
print(df_s24_only["panel"].value_counts())

print("\nS24-only cells by window_id:")
print(df_s24_only.groupby(["panel", "window_id"]).size().sort_values(ascending=False).head(15))
