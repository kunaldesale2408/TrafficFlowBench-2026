"""Generate Task 2 High-Confidence Physical Hole Closure Queue:
Takes the champion S33 Queue (10,596 positives) and closes only high-confidence
1-link topological shockwave gaps along mainline highway corridors:
Pattern: link[i-1] == 1, link[i] == 0, link[i+1] == 1
Condition: continuous dual model probability p_s24 >= 0.40

Outputs:
  outputs/submissions/queue_s42_closure.csv (174,000 rows)
"""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("trafficflowbench").resolve()
DATA = ROOT / "data" / "kaggle_public"
SUB_S33 = ROOT / "outputs" / "submissions" / "submission_s33.csv"
PROB_FILE = ROOT / "scratch" / "test_ongoing_probabilities.parquet"
OUT_CSV = ROOT / "outputs" / "submissions" / "queue_s42_closure.csv"


def main() -> None:
    t0 = time.time()
    print("=" * 75)
    print(">>> GENERATING HIGH-CONFIDENCE HOLE CLOSURE QUEUE (TASK 2) <<<")
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

    # 3. Load S33 champion queue
    print(f"Loading champion S33 queue from {SUB_S33}...")
    sub_s33 = pd.read_csv(SUB_S33)
    q_base = sub_s33.loc[qu_mask, "queue_pred"].to_numpy().astype(int)
    key_qu["pred_orig"] = q_base
    key_qu["queue_pred"] = q_base.copy()

    # 4. Load continuous probabilities
    print(f"Loading ongoing model probabilities from {PROB_FILE}...")
    probs = pd.read_parquet(PROB_FILE)
    prob_keys = ["window_id", "h", "link_id"]
    for k in prob_keys:
        probs[k] = probs[k].astype(str)
        key_qu[k] = key_qu[k].astype(str)

    merged = key_qu.merge(probs[prob_keys + ["p_s24"]], on=prob_keys, how="left")

    # 5. Identify and bridge topological holes [1, 0, 1] with p_s24 >= 0.40
    print("Scanning corridors for physical shockwave gaps [1, 0, 1]...")
    bridged_indices = []

    for panel in wi_df["panel"].unique():
        topo_path = DATA / "corridors" / panel / "network" / "lwr_mainline_topology.csv"
        if topo_path.exists():
            topo = pd.read_csv(topo_path)
            ordered_links = topo["link_id"].astype(str).tolist()

            p_df = merged[merged["panel"] == panel]
            for (w_id, h), grp in p_df.groupby(["window_id", "h"]):
                pred_map = dict(zip(grp["link_id"], grp["pred_orig"]))
                prob_map = dict(zip(grp["link_id"], grp["p_s24"]))
                idx_map = dict(zip(grp["link_id"], grp.index))

                arr = np.array([pred_map.get(l, 0) for l in ordered_links])
                hole_mask = (arr[:-2] == 1) & (arr[1:-1] == 0) & (arr[2:] == 1)
                if hole_mask.any():
                    indices = np.where(hole_mask)[0] + 1
                    for idx in indices:
                        link = ordered_links[idx]
                        p_val = prob_map.get(link, 0.0)
                        if p_val >= 0.40:
                            bridged_indices.append(idx_map[link])

    print(f"Found {len(bridged_indices)} high-confidence physical holes to bridge.")
    merged.loc[bridged_indices, "queue_pred"] = 1

    merged["split"] = np.where(merged["timestamp"].str[:7] == "2031-03", "validation (LB)", "private")
    print("\nPositives by Split:")
    print(merged.groupby("split")[["pred_orig", "queue_pred"]].sum())

    # 6. Verify integrity and save
    out_df = merged[["window_id", "timestamp", "link_id", "queue_pred"]].copy()
    assert len(out_df) == 174_000, f"Expected 174,000 queue rows, got {len(out_df)}"
    assert set(out_df["queue_pred"].unique()).issubset({0, 1}), "Non-binary predictions!"
    assert out_df["queue_pred"].isna().sum() == 0, "NaNs in predictions!"

    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved queue to {OUT_CSV} in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
