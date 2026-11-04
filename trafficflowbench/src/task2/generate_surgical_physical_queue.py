"""Generate Task 2 Surgical Physical Queue Forecast:
Applies high-precision physical corrections on top of the 0.87141 Champion Queue:
1. Promote indisputable false negatives (87=0, p_s24 >= 0.52, v_T <= 45.0 km/h).
2. Clear indisputable false positives (87=1, p_s24 <= 0.25, v_T >= 95.0 km/h).
Total adjustments: 83 promotions, 79 clearances (preserving the optimal sensitivity balance).

Outputs:
  outputs/submissions/queue_s41_surgical.csv (174,000 rows)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("trafficflowbench").resolve()
DATA = ROOT / "data" / "kaggle_public"
SUB_87 = ROOT / "kaggle_research" / "semwal_087" / "output" / "submission.csv"
PROB_FILE = ROOT / "scratch" / "test_ongoing_probabilities.parquet"
OUT_CSV = ROOT / "outputs" / "submissions" / "queue_s41_surgical.csv"


def main() -> None:
    t0 = time.time()
    print("=" * 75)
    print(">>> GENERATING SURGICAL PHYSICAL QUEUE FORECASTS (TASK 2) <<<")
    print("=" * 75)

    # 1. Load submission key
    print("Loading submission key...")
    key = pd.read_csv(DATA / "submission_key.csv", low_memory=False)
    qu_mask = key["task"] == "queue"
    key_qu = key[qu_mask].copy()

    # 2. Map window conditions and forecast origin
    print("Loading window metadata...")
    wi_files = list((DATA / "task2").glob("*/*/window_index.csv"))
    wi_df = pd.concat([pd.read_csv(f) for f in wi_files], ignore_index=True)
    cond_map = dict(zip(wi_df["window_id"].astype(str), wi_df["condition"].astype(str)))
    origin_map = dict(zip(wi_df["window_id"].astype(str), wi_df["forecast_origin"].astype(str)))

    key_qu["condition"] = key_qu["window_id"].astype(str).map(cond_map)
    key_qu["forecast_origin"] = key_qu["window_id"].astype(str).map(origin_map)
    key_qu["h"] = ((pd.to_datetime(key_qu["timestamp"], utc=True) - pd.to_datetime(key_qu["forecast_origin"], utc=True)).dt.total_seconds() // 300).astype(int)

    # 3. Load 0.87141 champion queue
    print(f"Loading champion queue baseline from {SUB_87}...")
    sub_87 = pd.read_csv(SUB_87)
    q_base = sub_87.loc[qu_mask, "queue_pred"].to_numpy().astype(int)
    key_qu["pred_orig"] = q_base
    key_qu["queue_pred"] = q_base.copy()

    # 4. Load continuous ongoing probabilities
    print(f"Loading ongoing model probabilities from {PROB_FILE}...")
    probs = pd.read_parquet(PROB_FILE)
    prob_keys = ["window_id", "h", "link_id"]
    for k in prob_keys:
        probs[k] = probs[k].astype(str)
        key_qu[k] = key_qu[k].astype(str)

    merged = key_qu.merge(probs[prob_keys + ["p_s24"]], on=prob_keys, how="left")

    # 5. Extract observed speed at origin T from window_history
    print("Extracting physical speed at forecast origin T from window_history...")
    v_T_cache = {}
    for split in ("validation", "private"):
        for panel in wi_df["panel"].unique():
            f = DATA / "task2" / panel / split / "window_history.parquet"
            if f.exists():
                h_df = pd.read_parquet(f)
                h_df["link_id"] = h_df["link_id"].astype(str)
                for w_id, w_grp in h_df.groupby("window_id"):
                    max_t = w_grp["timestamp"].max()
                    t_slice = w_grp[w_grp["timestamp"] == max_t]
                    for row in t_slice.itertuples():
                        v_T_cache[(str(w_id), str(row.link_id))] = row.speed_kmh

    merged["v_T"] = [v_T_cache.get((w, l), np.nan) for w, l in zip(merged["window_id"], merged["link_id"])]

    # 6. Apply Surgical Physical Corrections on Ongoing Windows
    is_ong = merged["condition"] == "queue_ongoing"
    promote = is_ong & (merged["pred_orig"] == 0) & (merged["p_s24"] >= 0.52) & (merged["v_T"] <= 45.0)
    clear = is_ong & (merged["pred_orig"] == 1) & (merged["p_s24"] <= 0.25) & (merged["v_T"] >= 95.0)

    merged.loc[promote, "queue_pred"] = 1
    merged.loc[clear, "queue_pred"] = 0

    n_promoted = int(promote.sum())
    n_cleared = int(clear.sum())
    print("\n" + "=" * 60)
    print(">>> SURGICAL PHYSICAL CORRECTION RESULTS <<<")
    print("=" * 60)
    print(f"Total Queue Cells:      {len(merged):,}")
    print(f"Original Positives:     {q_base.sum():,} ({q_base.mean()*100:.2f}%)")
    print(f"Surgically Promoted:    +{n_promoted} cells (v_T <= 45 km/h, p_s24 >= 0.52)")
    print(f"Surgically Cleared:     -{n_cleared} cells (v_T >= 95 km/h, p_s24 <= 0.25)")
    print(f"Final Positives:        {merged['queue_pred'].sum():,} ({merged['queue_pred'].mean()*100:.2f}%)")

    merged["split"] = np.where(merged["timestamp"].str[:7] == "2031-03", "validation (LB)", "private")
    print("\nPositives by Split:")
    print(merged.groupby("split")[["pred_orig", "queue_pred"]].sum())

    # 7. Verify integrity and save
    out_df = merged[["window_id", "timestamp", "link_id", "queue_pred"]].copy()
    assert len(out_df) == 174_000, f"Expected 174,000 rows, got {len(out_df)}"
    assert set(out_df["queue_pred"].unique()).issubset({0, 1}), "Non-binary predictions!"
    assert out_df["queue_pred"].isna().sum() == 0, "NaNs in predictions!"

    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved surgical physical queue to {OUT_CSV} in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
