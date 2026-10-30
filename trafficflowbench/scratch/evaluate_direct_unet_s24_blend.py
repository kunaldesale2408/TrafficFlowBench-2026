"""Evaluate Direct U-Net/Trees + S24 Champion State Blend:
Directly combines:
  - 50% Semwal Pure U-Net + Trees (task1_validation.csv + task1_private.csv)
  - 50% S24 Champion Multi-Seed Deep-LGB + CatBoost (state_s24_ensemble_fd.csv)
Comparing against the 0.87141 state (which used raw Anubis CatBoost instead of S24).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("trafficflowbench").resolve()
T1_DIR = ROOT / "kaggle_research" / "semwal_087" / "output" / "task1"
SUB_87_PATH = ROOT / "kaggle_research" / "semwal_087" / "output" / "submission.csv"
S24_STATE_PATH = ROOT / "outputs" / "submissions" / "state_s24_ensemble_fd.csv"
KEY_PATH = ROOT / "data" / "kaggle_public" / "submission_key.csv"

print("Loading Semwal pure Task 1 predictions...")
t1_val = pd.read_csv(T1_DIR / "task1_validation.csv")
t1_priv = pd.read_csv(T1_DIR / "task1_private.csv")
semwal_raw = pd.concat([t1_val, t1_priv], ignore_index=True)
print(f"Semwal raw state rows: {len(semwal_raw):,}")

print("Loading S24 Champion state...")
s24_state = pd.read_csv(S24_STATE_PATH)
print(f"S24 state rows: {len(s24_state):,}")

print("Loading 0.87141 master submission...")
sub_87 = pd.read_csv(SUB_87_PATH)
st_mask = sub_87["task"] == "state"
sub_87_st = sub_87[st_mask].copy()

# Match ordering via submission_key
key = pd.read_csv(KEY_PATH)
key_st = key[key["task"] == "state"].copy()

print("Verifying alignment...")
# Merge Semwal raw by keys
keys = ["panel", "timestamp", "station_id", "link_id", "mask_regime"]
for k in keys:
    semwal_raw[k] = semwal_raw[k].astype(str)
    s24_state[k] = s24_state[k].astype(str)
    key_st[k] = key_st[k].astype(str)

m_sem = key_st.merge(semwal_raw[keys + ["speed_kmh", "flow_vph"]], on=keys, how="left")
m_s24 = key_st.merge(s24_state[keys + ["speed_kmh", "flow_vph"]], on=keys, how="left")

assert m_sem["speed_kmh"].notna().all(), "Missing speed in Semwal merge!"
assert m_s24["speed_kmh"].notna().all(), "Missing speed in S24 merge!"

v_sem = m_sem["speed_kmh"].to_numpy()
q_sem = m_sem["flow_vph"].to_numpy()
v_s24 = m_s24["speed_kmh"].to_numpy()
q_s24 = m_s24["flow_vph"].to_numpy()

v_direct_blend = 0.50 * v_sem + 0.50 * v_s24
q_direct_blend = 0.50 * q_sem + 0.50 * q_s24

v_87 = sub_87_st["speed_kmh"].to_numpy()
q_87 = sub_87_st["flow_vph"].to_numpy()

print("\n" + "=" * 70)
print(">>> DIRECT BLEND (SEMWAL U-NET + S24) vs 0.87141 BENCHMARK STATE <<<")
print("=" * 70)

diff_v = v_direct_blend - v_87
diff_q = q_direct_blend - q_87

print(f"Speed difference vs 0.87141: mean = {np.mean(diff_v):.4f} km/h, std = {np.std(diff_v):.4f} km/h, RMSE = {np.sqrt(np.mean(diff_v**2)):.4f} km/h")
print(f"Flow difference vs 0.87141:  mean = {np.mean(diff_q):.2f} vph,  std = {np.std(diff_q):.2f} vph,  RMSE = {np.sqrt(np.mean(diff_q**2)):.2f} vph")

print(f"\nDirect Blend Speed: min={v_direct_blend.min():.2f}, mean={v_direct_blend.mean():.2f}, max={v_direct_blend.max():.2f}")
print(f"0.87141 Speed:      min={v_87.min():.2f}, mean={v_87.mean():.2f}, max={v_87.max():.2f}")

print(f"\nDirect Blend Flow:  min={q_direct_blend.min():.1f}, mean={q_direct_blend.mean():.1f}, max={q_direct_blend.max():.1f}")
print(f"0.87141 Flow:       min={q_87.min():.1f}, mean={q_87.mean():.1f}, max={q_87.max():.1f}")

# Physical checks
print(f"\nAll speeds >= 5 km/h: {(v_direct_blend >= 5.0).all()}")
print(f"All flows >= 50 vph:  {(q_direct_blend >= 50.0).all()}")
