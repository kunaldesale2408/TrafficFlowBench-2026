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
key_qu["h"] = ((pd.to_datetime(key_qu["timestamp"], utc=True) - pd.to_datetime(key_qu["forecast_origin"], utc=True)).dt.total_seconds() // 300).astype(int)
key_qu["split"] = np.where(key_qu["timestamp"].str[:7] == "2031-03", "validation", "private")

# 2. Load S44 queue
s44 = pd.read_csv(ROOT / "outputs" / "submissions" / "queue_s44_precision.csv")
merged = key_qu.merge(s44, on=["window_id", "timestamp", "link_id"], how="inner").rename(columns={"queue_pred": "s44"})

# 3. Load Semwal queue
sem_val = pd.read_csv(ROOT / "kaggle_research" / "semwal_087" / "output" / "task2" / "task2_validation.csv")
sem_priv = pd.read_csv(ROOT / "kaggle_research" / "semwal_087" / "output" / "task2" / "task2_private.csv")
sem_all = pd.concat([sem_val, sem_priv], ignore_index=True)
merged = merged.merge(sem_all, on=["window_id", "timestamp", "link_id"], how="inner").rename(columns={"queue_pred": "sem"})

# 4. Load probabilities
probs = pd.read_parquet(ROOT / "scratch" / "test_ongoing_probabilities.parquet")
prob_keys = ["window_id", "h", "link_id"]
for k in prob_keys:
    probs[k] = probs[k].astype(str)
    merged[k] = merged[k].astype(str)
merged = merged.merge(probs[prob_keys + ["p_s24", "p_lgb", "p_cb"]], on=prob_keys, how="left")

print("\n--- DISCORDANT ONGOING CELLS ANALYSIS (VALIDATION) ---")
m_val = merged[(merged["split"] == "validation") & (merged["condition"] == "queue_ongoing")]
s44_only = m_val[(m_val["s44"] == 1) & (m_val["sem"] == 0)]
sem_only = m_val[(m_val["s44"] == 0) & (m_val["sem"] == 1)]

print(f"S44=1, Semwal=0 (132 cells):")
print(f"  p_s24 distribution: min={s44_only['p_s24'].min():.3f}, mean={s44_only['p_s24'].mean():.3f}, median={s44_only['p_s24'].median():.3f}, max={s44_only['p_s24'].max():.3f}")
print(f"  h distribution:\n{s44_only['h'].value_counts().sort_index()}")
print(f"  panels:\n{s44_only['panel'].value_counts()}")

print(f"\nSemwal=1, S44=0 (233 cells):")
print(f"  p_s24 distribution: min={sem_only['p_s24'].min():.3f}, mean={sem_only['p_s24'].mean():.3f}, median={sem_only['p_s24'].median():.3f}, max={sem_only['p_s24'].max():.3f}")
print(f"  h distribution:\n{sem_only['h'].value_counts().sort_index()}")
print(f"  panels:\n{sem_only['panel'].value_counts()}")

