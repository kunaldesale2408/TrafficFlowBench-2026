"""Verify and evaluate blending S24 (0.86760) and Benchmark 0.87141:
Task 1: 50% S24 + 50% 0.87141
Task 2: 0.87141 queue (10,596 positives, identical onset, validated 0.87141 ongoing)
Task 4: Reference exact solve
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("trafficflowbench").resolve()
SUB_87141 = ROOT / "kaggle_research" / "semwal_087" / "output" / "submission.csv"
SUB_S24 = ROOT / "outputs" / "submissions" / "submission_s24.csv"

print("Loading submissions...")
df_87 = pd.read_csv(SUB_87141)
df_s24 = pd.read_csv(SUB_S24)

st = df_87["task"] == "state"
qu = df_87["task"] == "queue"
od = df_87["task"] == "odme"

# 1. State 50/50 blend
v_blend = 0.5 * df_87.loc[st, "speed_kmh"].to_numpy() + 0.5 * df_s24.loc[st, "speed_kmh"].to_numpy()
q_blend = 0.5 * df_87.loc[st, "flow_vph"].to_numpy() + 0.5 * df_s24.loc[st, "flow_vph"].to_numpy()

print("\n--- Blended State Properties ---")
print(f"Speed: min={v_blend.min():.2f}, mean={v_blend.mean():.2f}, max={v_blend.max():.2f}")
print(f"Flow:  min={q_blend.min():.1f}, mean={q_blend.mean():.1f}, max={q_blend.max():.1f}")
print(f"All finite: {np.isfinite(v_blend).all() and np.isfinite(q_blend).all()}")
print(f"Min speed >= 5 km/h: {(v_blend >= 5.0).all()}")
print(f"Min flow >= 50 vph: {(q_blend >= 50.0).all()}")

# 2. Queue properties
q_pred = df_87.loc[qu, "queue_pred"].to_numpy()
print("\n--- Queue Properties ---")
print(f"Total positive queue cells: {q_pred.sum():,} ({q_pred.mean()*100:.2f}%)")
print(f"Binary {0, 1}: {set(np.unique(q_pred)) <= {0, 1}}")

# 3. ODME properties
od_flow = df_s24.loc[od, "path_flow"].to_numpy()
print("\n--- ODME Properties ---")
print(f"Path flow: min={od_flow.min():.3f}, mean={od_flow.mean():.3f}, max={od_flow.max():.3f}")
print(f"All non-negative: {(od_flow >= 0).all()}")
