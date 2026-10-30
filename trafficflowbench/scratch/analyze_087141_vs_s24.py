"""Detailed Task-by-Task Diagnostic Comparison:
0.87141 Benchmark Submission vs S24 Personal Best (0.86760)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("trafficflowbench").resolve()
SUB_87141 = ROOT / "kaggle_research" / "semwal_087" / "output" / "submission.csv"
SUB_S24 = ROOT / "outputs" / "submissions" / "submission_s24.csv"
KEY_FILE = ROOT / "data" / "kaggle_public" / "submission_key.csv"

print("Loading 0.87141 submission...")
df_87 = pd.read_csv(SUB_87141)
print(f"Loaded 87: {len(df_87):,} rows.")

print("Loading S24 submission...")
df_s24 = pd.read_csv(SUB_S24)
print(f"Loaded S24: {len(df_s24):,} rows.")

print("Loading submission key...")
key = pd.read_csv(KEY_FILE)

# 1. Overall Task distribution
print("\n" + "=" * 70)
print(">>> TASK DISTRIBUTION COMPARISON <<<")
print("=" * 70)
for task in ["state", "queue", "odme"]:
    mask = df_87["task"] == task
    print(f"Task {task.upper()}: {mask.sum():,} rows")

# 2. Task 1: State Reconstruction
print("\n" + "=" * 70)
print(">>> TASK 1: STATE COMPARISON <<<")
print("=" * 70)
st = df_87["task"] == "state"
speed_diff = df_87.loc[st, "speed_kmh"].to_numpy() - df_s24.loc[st, "speed_kmh"].to_numpy()
flow_diff = df_87.loc[st, "flow_vph"].to_numpy() - df_s24.loc[st, "flow_vph"].to_numpy()

print(f"Speed difference: mean = {np.mean(speed_diff):.3f} km/h, std = {np.std(speed_diff):.3f} km/h, RMSE = {np.sqrt(np.mean(speed_diff**2)):.3f} km/h")
print(f"  0.87141 Speed: min={df_87.loc[st, 'speed_kmh'].min():.2f}, mean={df_87.loc[st, 'speed_kmh'].mean():.2f}, max={df_87.loc[st, 'speed_kmh'].max():.2f}")
print(f"  S24 Speed:     min={df_s24.loc[st, 'speed_kmh'].min():.2f}, mean={df_s24.loc[st, 'speed_kmh'].mean():.2f}, max={df_s24.loc[st, 'speed_kmh'].max():.2f}")

print(f"\nFlow difference: mean = {np.mean(flow_diff):.2f} vph, std = {np.std(flow_diff):.2f} vph, RMSE = {np.sqrt(np.mean(flow_diff**2)):.2f} vph")
print(f"  0.87141 Flow: min={df_87.loc[st, 'flow_vph'].min():.1f}, mean={df_87.loc[st, 'flow_vph'].mean():.1f}, max={df_87.loc[st, 'flow_vph'].max():.1f}")
print(f"  S24 Flow:     min={df_s24.loc[st, 'flow_vph'].min():.1f}, mean={df_s24.loc[st, 'flow_vph'].mean():.1f}, max={df_s24.loc[st, 'flow_vph'].max():.1f}")

# 3. Task 2: Queue Forecasting
print("\n" + "=" * 70)
print(">>> TASK 2: QUEUE COMPARISON <<<")
print("=" * 70)
qu = df_87["task"] == "queue"
q_87 = df_87.loc[qu, "queue_pred"].to_numpy().astype(int)
q_s24 = df_s24.loc[qu, "queue_pred"].to_numpy().astype(int)

agree = (q_87 == q_s24).mean()
diff_cells = (q_87 != q_s24).sum()
pos_87 = q_87.sum()
pos_s24 = q_s24.sum()
intersection = ((q_87 == 1) & (q_s24 == 1)).sum()
union = ((q_87 == 1) | (q_s24 == 1)).sum()
iou = intersection / max(union, 1)

print(f"Queue cell agreement: {agree * 100:.3f}% ({diff_cells:,} differing cells out of {len(q_87):,})")
print(f"  0.87141 Positives: {pos_87:,} ({pos_87 / len(q_87) * 100:.2f}%)")
print(f"  S24 Positives:     {pos_s24:,} ({pos_s24 / len(q_s24) * 100:.2f}%)")
print(f"  Direct IoU overlap: {iou:.4f} (intersection={intersection:,}, union={union:,})")

# Breakdown by condition (ongoing vs onset)
k_qu = key[key["task"] == "queue"].copy()
wi_files = list((ROOT / "data" / "kaggle_public" / "task2").glob("*/*/window_index.csv"))
wi_df = pd.concat([pd.read_csv(f) for f in wi_files], ignore_index=True)
cond_map = dict(zip(wi_df["window_id"].astype(str), wi_df["condition"].astype(str)))
k_qu["condition"] = k_qu["window_id"].astype(str).map(cond_map)

is_ongoing = (k_qu["condition"] == "queue_ongoing").to_numpy()
is_onset = (k_qu["condition"] == "queue_onset").to_numpy()

print(f"\n  Ongoing windows ({is_ongoing.sum():,} cells):")
print(f"    0.87141 Positives: {q_87[is_ongoing].sum():,} ({q_87[is_ongoing].mean()*100:.2f}%)")
print(f"    S24 Positives:     {q_s24[is_ongoing].sum():,} ({q_s24[is_ongoing].mean()*100:.2f}%)")
print(f"    Agreement:         {(q_87[is_ongoing] == q_s24[is_ongoing]).mean()*100:.3f}%")
print(f"    Differing cells:   {(q_87[is_ongoing] != q_s24[is_ongoing]).sum():,}")

print(f"\n  Onset windows ({is_onset.sum():,} cells):")
print(f"    0.87141 Positives: {q_87[is_onset].sum():,} ({q_87[is_onset].mean()*100:.2f}%)")
print(f"    S24 Positives:     {q_s24[is_onset].sum():,} ({q_s24[is_onset].mean()*100:.2f}%)")
print(f"    Agreement:         {(q_87[is_onset] == q_s24[is_onset]).mean()*100:.3f}%")
print(f"    Differing cells:   {(q_87[is_onset] != q_s24[is_onset]).sum():,}")

# 4. Task 4: ODME Comparison
print("\n" + "=" * 70)
print(">>> TASK 4: ODME COMPARISON <<<")
print("=" * 70)
od = df_87["task"] == "odme"
od_diff = df_87.loc[od, "path_flow"].to_numpy() - df_s24.loc[od, "path_flow"].to_numpy()
print(f"Max absolute path_flow difference: {np.max(np.abs(od_diff)):.6f}")
print(f"Mean path_flow difference:         {np.mean(np.abs(od_diff)):.6f}")
print(f"Agreement (> 1e-4):                {(np.abs(od_diff) < 1e-4).mean()*100:.2f}%")
