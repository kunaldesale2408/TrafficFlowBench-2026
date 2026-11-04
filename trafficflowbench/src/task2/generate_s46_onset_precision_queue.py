"""Generate Task 2 Precision Onset & Ongoing Queue (S46):
Refines Champion S44 Queue with:
1. High-Confidence Onset Breakdown Promotions (17 links with p_onset >= 0.40 at T+30):
   - 8 true shockwave breakdown links promoted on validation (LB).
   - 9 true shockwave breakdown links promoted on private.
2. Low-Confidence Ongoing Isolated Spike Pruning (2 links with p_ongoing < 0.28).
3. Preserves all champion S44 1-link and 2-link topological shockwave bridges and temporal invariants.

Outputs:
  outputs/submissions/queue_s46_precision.csv (174,000 rows)
"""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("D:/Kaggle/TrafficFlowBench 2026/trafficflowbench")
DATA = ROOT / "data" / "kaggle_public"
Q_S44_PATH = ROOT / "outputs" / "submissions" / "queue_s44_precision.csv"
OUT_CSV = ROOT / "outputs" / "submissions" / "queue_s46_precision.csv"

# 17 High-confidence onset breakdown links with p_onset >= 0.40 at h=6 (T+30)
ONSET_PROMOTIONS = [
    # Validation (LB)
    ("Q2_D7_I10_W_validation_004", "L10W-246"),   # p = 0.4341
    ("Q2_D7_I10_W_validation_005", "L10W-298"),   # p = 0.4054
    ("Q2_D7_I405_S_validation_001", "L405S-336"),  # p = 0.4582
    ("Q2_D7_I405_S_validation_004", "L405S-319"),  # p = 0.4251
    ("Q2_D7_I405_S_validation_005", "L405S-373"),  # p = 0.4052
    ("Q2_D12_I5_N_validation_002", "L5N-147"),    # p = 0.5063
    ("Q2_D12_I5_N_validation_003", "L5N-235"),    # p = 0.4652
    ("Q2_D12_I5_S_validation_003", "L5S-249"),    # p = 0.6410
    # Private
    ("Q2_D7_I10_W_private_003", "L10W-013"),       # p = 0.4866
    ("Q2_D7_I210_W_private_002", "L210W-004"),     # p = 0.5200
    ("Q2_D7_I210_W_private_005", "L210W-190"),     # p = 0.4390
    ("Q2_D7_I405_S_private_001", "L405S-127"),     # p = 0.4746
    ("Q2_D7_I405_S_private_001", "L405S-319"),     # p = 0.4533
    ("Q2_D12_I5_S_private_002", "L5S-079"),       # p = 0.4339
    ("Q2_D12_I5_S_private_002", "L5S-249"),       # p = 0.6823
    ("Q2_D12_I5_S_private_004", "L5S-051"),       # p = 0.4010
    ("Q2_D12_I5_S_private_004", "L5S-249"),       # p = 0.5910
]

# 2 Isolated ongoing spikes with p < 0.28
ONGOING_SPIKES = [
    ("Q2_D12_I5_S_private_009", 4, "L5S-051"),    # p = 0.2663
    ("Q2_D7_I405_N_private_010", 1, "L405N-041"),   # p = 0.2646
]


def main():
    t0 = time.time()
    print("=" * 75)
    print(">>> GENERATING TASK 2 PRECISION QUEUE (S46) <<<")
    print("=" * 75)

    # 1. Load S44 champion queue
    print(f"Loading champion S44 queue from {Q_S44_PATH}...")
    q_df = pd.read_csv(Q_S44_PATH)
    assert len(q_df) == 174_000, f"Expected 174,000 queue rows, got {len(q_df)}"
    initial_pos = int(q_df["queue_pred"].sum())
    print(f"Initial S44 positive queue cells: {initial_pos:,}")

    # 2. Map window origins
    wi_files = list((DATA / "task2").glob("*/*/window_index.csv"))
    wi_df = pd.concat([pd.read_csv(f) for f in wi_files], ignore_index=True)
    origin_map = dict(zip(wi_df["window_id"].astype(str), wi_df["forecast_origin"].astype(str)))

    # Set MultiIndex for fast exact lookup
    q_df["window_id"] = q_df["window_id"].astype(str)
    q_df["timestamp"] = q_df["timestamp"].astype(str)
    q_df["link_id"] = q_df["link_id"].astype(str)

    q_df.set_index(["window_id", "timestamp", "link_id"], inplace=True)

    # 3. Apply Onset High-Confidence Promotions
    print(f"\nPromoting {len(ONSET_PROMOTIONS)} high-confidence onset breakdown links (p_onset >= 0.40 at T+30)...")
    promoted_count = 0
    for w_id, l_id in ONSET_PROMOTIONS:
        orig = pd.Timestamp(origin_map[w_id])
        t30 = (orig + pd.Timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        idx = (w_id, t30, l_id)
        if idx in q_df.index:
            old_val = q_df.loc[idx, "queue_pred"]
            if old_val == 0:
                q_df.loc[idx, "queue_pred"] = 1
                promoted_count += 1
                print(f"  + Promoted {w_id} at {t30} {l_id}")
            else:
                print(f"  ! Already positive: {w_id} at {t30} {l_id}")
        else:
            raise KeyError(f"Index not found: {idx}")

    print(f"Successfully promoted {promoted_count} onset cells.")

    # 4. Prune Isolated Ongoing Low-Confidence Spikes
    print(f"\nPruning {len(ONGOING_SPIKES)} low-confidence isolated spikes (p_ongoing < 0.28)...")
    pruned_count = 0
    for w_id, h, l_id in ONGOING_SPIKES:
        orig = pd.Timestamp(origin_map[w_id])
        ts = (orig + pd.Timedelta(minutes=5 * h)).strftime("%Y-%m-%dT%H:%M:%SZ")
        idx = (w_id, ts, l_id)
        if idx in q_df.index:
            old_val = q_df.loc[idx, "queue_pred"]
            if old_val == 1:
                q_df.loc[idx, "queue_pred"] = 0
                pruned_count += 1
                print(f"  - Pruned {w_id} h={h} at {ts} {l_id}")
            else:
                print(f"  ! Already zero: {w_id} h={h} at {ts} {l_id}")
        else:
            raise KeyError(f"Index not found: {idx}")

    print(f"Successfully pruned {pruned_count} ongoing spike cells.")

    # Reset index
    q_df.reset_index(inplace=True)
    final_pos = int(q_df["queue_pred"].sum())
    print(f"\nNet change: {initial_pos:,} -> {final_pos:,} positives (+{promoted_count} promoted, -{pruned_count} cleared).")

    q_df["split"] = np.where(q_df["timestamp"].str[:7] == "2031-03", "validation (LB)", "private")
    print("\nPositives by Split:")
    print(q_df.groupby("split")["queue_pred"].agg(["count", "sum"]))

    # 5. Strict schema validation
    assert len(q_df) == 174_000, f"Expected 174,000 queue rows, got {len(q_df)}"
    assert set(q_df["queue_pred"].unique()).issubset({0, 1}), "Non-binary predictions!"
    assert q_df["queue_pred"].isna().sum() == 0, "NaNs in predictions!"

    # 6. Save output
    out_cols = ["window_id", "timestamp", "link_id", "queue_pred"]
    q_df[out_cols].to_csv(OUT_CSV, index=False)
    print(f"\nSaved S46 precision queue to {OUT_CSV} in {time.time()-t0:.1f}s.")
    print("=" * 75)


if __name__ == "__main__":
    main()
