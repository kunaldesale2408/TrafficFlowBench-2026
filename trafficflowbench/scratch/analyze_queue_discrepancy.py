import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("trafficflowbench")
DATA = ROOT / "data" / "kaggle_public"

# 1. Load submission key & window index
key = pd.read_csv(DATA / "submission_key.csv", low_memory=False)
qu_mask = key["task"] == "queue"
key_qu = key[qu_mask].copy()

wi_files = list((DATA / "task2").glob("*/*/window_index.csv"))
wi_df = pd.concat([pd.read_csv(f) for f in wi_files], ignore_index=True)
cond_map = dict(zip(wi_df["window_id"].astype(str), wi_df["condition"].astype(str)))
origin_map = dict(zip(wi_df["window_id"].astype(str), wi_df["forecast_origin"].astype(str)))
panel_map = dict(zip(wi_df["window_id"].astype(str), wi_df["panel"].astype(str)))

key_qu["condition"] = key_qu["window_id"].map(cond_map)
key_qu["forecast_origin"] = key_qu["window_id"].map(origin_map)
key_qu["panel"] = key_qu["window_id"].map(panel_map)
key_qu["split"] = np.where(key_qu["timestamp"].str[:7] == "2031-03", "validation", "private")

# Filter to validation only (LB scored)
val_key = key_qu[key_qu["split"] == "validation"].copy()
print(f"Validation queue rows: {len(val_key):,}")

# 2. Load S44 queue
s44_df = pd.read_csv(ROOT / "outputs" / "submissions" / "queue_s44_precision.csv")
merged = val_key.merge(s44_df, on=["window_id", "timestamp", "link_id"], how="inner")
merged = merged.rename(columns={"queue_pred": "pred_s44"})

# 3. Load Semwal queue
sem_val_df = pd.read_csv(ROOT / "kaggle_research" / "semwal_087" / "output" / "task2" / "task2_validation.csv")
merged = merged.merge(sem_val_df, on=["window_id", "timestamp", "link_id"], how="inner")
merged = merged.rename(columns={"queue_pred": "pred_sem"})

print(f"Total merged validation cells: {len(merged):,}")

# Compare positives
print("\n--- POSITIVES BY CONDITION (VALIDATION) ---")
for cond, grp in merged.groupby("condition"):
    p44 = grp["pred_s44"].sum()
    p_sem = grp["pred_sem"].sum()
    inter = (grp["pred_s44"] & grp["pred_sem"]).sum()
    union = (grp["pred_s44"] | grp["pred_sem"]).sum()
    print(f"[{cond}]")
    print(f"  S44: {p44} | Semwal: {p_sem} | Overlap: {inter} | Union: {union} | IoU: {inter/union:.4f}")

# Look at onset windows specifically
print("\n--- ONSET WINDOWS BREAKDOWN (20 VALIDATION WINDOWS) ---")
onset_df = merged[merged["condition"] == "queue_onset"]
onset_w_summary = []
for wid, grp in onset_df.groupby("window_id"):
    pnl = grp["panel"].iloc[0]
    p44 = grp["pred_s44"].sum()
    p_sem = grp["pred_sem"].sum()
    inter = (grp["pred_s44"] & grp["pred_sem"]).sum()
    union = (grp["pred_s44"] | grp["pred_sem"]).sum()
    onset_w_summary.append({
        "window_id": wid,
        "panel": pnl,
        "S44_pos": p44,
        "Semwal_pos": p_sem,
        "Overlap": inter,
        "Diff": abs(p44 - p_sem)
    })

res_onset = pd.DataFrame(onset_w_summary)
print(res_onset.to_string(index=False))
