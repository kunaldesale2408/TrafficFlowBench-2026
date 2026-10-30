"""Analyze State Ensemble Statistics: Correlation, Differences, and Extremes
Between Semwal U-Net/Trees, S24 Multi-Seed Bagged Engine, and Anubis Baseline.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("trafficflowbench").resolve()
DATA = ROOT / "data" / "kaggle_public"
SEM_VAL = ROOT / "kaggle_research" / "semwal_087" / "output" / "task1" / "task1_validation.csv"
SEM_PRIV = ROOT / "kaggle_research" / "semwal_087" / "output" / "task1" / "task1_private.csv"
S24_STATE = ROOT / "outputs" / "submissions" / "state_s24_ensemble_fd.csv"
SUB_87 = ROOT / "kaggle_research" / "semwal_087" / "output" / "submission.csv"

print("Loading data...")
sem_val = pd.read_csv(SEM_VAL)
sem_priv = pd.read_csv(SEM_PRIV)
df_sem = pd.concat([sem_val, sem_priv], ignore_index=True)

df_s24 = pd.read_csv(S24_STATE)

sub_87 = pd.read_csv(SUB_87)
sub_87_st = sub_87[sub_87["task"] == "state"]

key = pd.read_csv(DATA / "submission_key.csv", low_memory=False)
key_st = key[key["task"] == "state"].copy()

keys = ["panel", "timestamp", "station_id", "link_id", "mask_regime"]
for k in keys:
    df_sem[k] = df_sem[k].astype(str)
    df_s24[k] = df_s24[k].astype(str)
    key_st[k] = key_st[k].astype(str)

m_sem = key_st.merge(df_sem[keys + ["speed_kmh", "flow_vph"]], on=keys, how="left")
m_s24 = key_st.merge(df_s24[keys + ["speed_kmh", "flow_vph"]], on=keys, how="left")

v_sem = m_sem["speed_kmh"].to_numpy()
q_sem = m_sem["flow_vph"].to_numpy()
v_s24 = m_s24["speed_kmh"].to_numpy()
q_s24 = m_s24["flow_vph"].to_numpy()
v_87 = sub_87_st["speed_kmh"].to_numpy()
q_87 = sub_87_st["flow_vph"].to_numpy()

# Anubis = 2 * sub_87 - semwal
v_anub = 2.0 * v_87 - v_sem
q_anub = 2.0 * q_87 - q_sem

print("\n" + "=" * 70)
print(">>> SPEED MODEL STATISTICS (6,735,795 test rows) <<<")
print("=" * 70)
print(f"Semwal Speed:  mean={v_sem.mean():.3f}, std={v_sem.std():.3f}, min={v_sem.min():.2f}, max={v_sem.max():.2f}")
print(f"S24 Speed:     mean={v_s24.mean():.3f}, std={v_s24.std():.3f}, min={v_s24.min():.2f}, max={v_s24.max():.2f}")
print(f"Anubis Speed:  mean={v_anub.mean():.3f}, std={v_anub.std():.3f}, min={v_anub.min():.2f}, max={v_anub.max():.2f}")

corr_v_sem_s24 = np.corrcoef(v_sem, v_s24)[0, 1]
corr_v_sem_anub = np.corrcoef(v_sem, v_anub)[0, 1]
corr_v_s24_anub = np.corrcoef(v_s24, v_anub)[0, 1]

print(f"\nSpeed Pearson Correlations:")
print(f"  Corr(Semwal, S24):    {corr_v_sem_s24:.6f}")
print(f"  Corr(Semwal, Anubis): {corr_v_sem_anub:.6f}")
print(f"  Corr(S24, Anubis):    {corr_v_s24_anub:.6f}")

print("\n" + "=" * 70)
print(">>> FLOW MODEL STATISTICS <<<")
print("=" * 70)
print(f"Semwal Flow:  mean={q_sem.mean():.1f}, std={q_sem.std():.1f}, min={q_sem.min():.1f}, max={q_sem.max():.1f}")
print(f"S24 Flow:     mean={q_s24.mean():.1f}, std={q_s24.std():.1f}, min={q_s24.min():.1f}, max={q_s24.max():.1f}")
print(f"Anubis Flow:  mean={q_anub.mean():.1f}, std={q_anub.std():.1f}, min={q_anub.min():.1f}, max={q_anub.max():.1f}")

corr_q_sem_s24 = np.corrcoef(q_sem, q_s24)[0, 1]
corr_q_sem_anub = np.corrcoef(q_sem, q_anub)[0, 1]
corr_q_s24_anub = np.corrcoef(q_s24, q_anub)[0, 1]

print(f"\nFlow Pearson Correlations:")
print(f"  Corr(Semwal, S24):    {corr_q_sem_s24:.6f}")
print(f"  Corr(Semwal, Anubis): {corr_q_sem_anub:.6f}")
print(f"  Corr(S24, Anubis):    {corr_q_s24_anub:.6f}")

# Pairwise Differences
diff_v_s24_sem = v_s24 - v_sem
diff_v_s24_anub = v_s24 - v_anub
print(f"\nSpeed Differences:")
print(f"  RMSE(S24 - Semwal): {np.sqrt(np.mean(diff_v_s24_sem**2)):.4f} km/h")
print(f"  RMSE(S24 - Anubis): {np.sqrt(np.mean(diff_v_s24_anub**2)):.4f} km/h")
print(f"  RMSE(Semwal - Anubis): {np.sqrt(np.mean((v_sem - v_anub)**2)):.4f} km/h")

diff_q_s24_sem = q_s24 - q_sem
diff_q_s24_anub = q_s24 - q_anub
print(f"\nFlow Differences:")
print(f"  RMSE(S24 - Semwal): {np.sqrt(np.mean(diff_q_s24_sem**2)):.2f} vph")
print(f"  RMSE(S24 - Anubis): {np.sqrt(np.mean(diff_q_s24_anub**2)):.2f} vph")
print(f"  RMSE(Semwal - Anubis): {np.sqrt(np.mean((q_sem - q_anub)**2)):.2f} vph")
