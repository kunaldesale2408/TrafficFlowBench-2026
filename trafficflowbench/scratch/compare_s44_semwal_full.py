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
panel_map = dict(zip(wi_df["window_id"].astype(str), wi_df["panel"].astype(str)))

key_qu["condition"] = key_qu["window_id"].map(cond_map)
key_qu["panel"] = key_qu["window_id"].map(panel_map)
key_qu["split"] = np.where(key_qu["timestamp"].str[:7] == "2031-03", "validation", "private")

# 2. Load S44 queue
s44 = pd.read_csv(ROOT / "outputs" / "submissions" / "queue_s44_precision.csv")
merged = key_qu.merge(s44, on=["window_id", "timestamp", "link_id"], how="inner").rename(columns={"queue_pred": "s44"})

# 3. Load Semwal queue (validation + private)
sem_val = pd.read_csv(ROOT / "kaggle_research" / "semwal_087" / "output" / "task2" / "task2_validation.csv")
sem_priv = pd.read_csv(ROOT / "kaggle_research" / "semwal_087" / "output" / "task2" / "task2_private.csv")
sem_all = pd.concat([sem_val, sem_priv], ignore_index=True)
merged = merged.merge(sem_all, on=["window_id", "timestamp", "link_id"], how="inner").rename(columns={"queue_pred": "sem"})

print(f"Total merged queue rows: {len(merged):,}")

for split in ["validation", "private"]:
    print(f"\n{'='*30} SPLIT: {split.upper()} {'='*30}")
    m_sp = merged[merged["split"] == split]
    for cond in ["queue_ongoing", "queue_onset"]:
        sub = m_sp[m_sp["condition"] == cond]
        s44_sum = sub["s44"].sum()
        sem_sum = sub["sem"].sum()
        inter = (sub["s44"] & sub["sem"]).sum()
        union = (sub["s44"] | sub["sem"]).sum()
        only_s44 = (sub["s44"] & ~sub["sem"]).sum()
        only_sem = (~sub["s44"] & sub["sem"]).sum()
        iou = inter / union if union > 0 else 1.0
        print(f"[{cond}]")
        print(f"  S44 positives: {s44_sum} | Semwal positives: {sem_sum}")
        print(f"  Overlap (agreement): {inter} ({inter/max(union,1):.4f} IoU)")
        print(f"  Only in S44: {only_s44} | Only in Semwal: {only_sem}")

