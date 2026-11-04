"""Inference Pipeline for Next-Gen Spatio-Temporal Shockwave Queue Forecaster.

Loads:
  outputs/models/nextgen_queue_lgb_s1.txt
  outputs/models/nextgen_queue_lgb_s2.txt
  outputs/models/nextgen_queue_lgb_s3.txt
  outputs/models/nextgen_queue_cb.cbm

Generates predictions for validation and private test windows (174,000 rows).
Applies high-confidence topological shockwave hole closure (p >= 0.40)
and isolated noise blip suppression (p <= 0.15).

Outputs:
  outputs/submissions/queue_nextgen_candidate.csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

# Import feature extractor from training script
import sys
PROJECT_ROOT = Path("trafficflowbench").resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src" / "task2"))
from train_nextgen_spatiotemporal_queue import STPanel, panel_names, HISTORY, HORIZON, THREADS

DATA = PROJECT_ROOT / "data" / "kaggle_public"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
OUT_CSV = PROJECT_ROOT / "outputs" / "submissions" / "queue_nextgen_candidate.csv"


def load_models():
    print("Loading Next-Gen Queue Ensemble models...")
    lgb_files = [
        MODELS_DIR / "nextgen_queue_lgb_s1.txt",
        MODELS_DIR / "nextgen_queue_lgb_s2.txt",
        MODELS_DIR / "nextgen_queue_lgb_s3.txt",
    ]
    boosters = [lgb.Booster(model_file=str(f), params={"num_threads": THREADS}) for f in lgb_files]
    print(f"Loaded {len(boosters)} LightGBM boosters.")

    cb_path = MODELS_DIR / "nextgen_queue_cb.cbm"
    cb = CatBoostClassifier()
    cb.load_model(str(cb_path))
    print("Loaded CatBoost classifier.")
    return boosters, cb


def predict_test_windows(boosters, cb, tau: float = 0.350, onset_fallback_k: int = 2) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print(">>> RUNNING INFERENCE ON VALIDATION & PRIVATE WINDOWS <<<")
    print("=" * 70)
    t0 = time.time()

    panels = panel_names()
    records = []

    for split in ("validation", "private"):
        for p_no, p_name in enumerate(panels):
            panel = STPanel(p_name, p_no)
            folder = DATA / "task2" / p_name / split
            wi_path = folder / "window_index.csv"
            hist_path = folder / "window_history.parquet"
            tpl_path = folder / "sample_submission_queue.csv"

            if not wi_path.exists():
                continue

            wi = pd.read_csv(wi_path, dtype={"window_id": str})
            hist = pd.read_parquet(hist_path)
            hist["window_id"] = hist["window_id"].astype(str)
            hist["link_id"] = hist["link_id"].astype(str)
            hist["timestamp"] = pd.to_datetime(hist["timestamp"], utc=True)

            print(f"  [{split} - {p_name}]: {len(wi)} windows")

            for row in wi.itertuples(index=False):
                wid = str(row.window_id)
                cond = str(row.condition)
                orig = pd.Timestamp(row.forecast_origin)

                times = pd.date_range(end=orig - pd.Timedelta(minutes=5), periods=HISTORY, freq="5min", tz="UTC")
                target_times = [orig + pd.Timedelta(minutes=5 * h) for h in range(1, HORIZON + 1)]

                w_hist = hist[hist["window_id"] == wid].copy()
                
                # Check causality
                assert w_hist["timestamp"].max() < orig, f"Leakage detected in {wid}!"

                idx = pd.Index(times)
                sp = w_hist.pivot(index="timestamp", columns="link_id", values="speed_kmh").reindex(index=idx, columns=panel.links).to_numpy(dtype=np.float32)
                fl = w_hist.pivot(index="timestamp", columns="link_id", values="flow_vph").reindex(index=idx, columns=panel.links).to_numpy(dtype=np.float32)
                el = w_hist.pivot(index="timestamp", columns="link_id", values="is_score_eligible").reindex(index=idx, columns=panel.links).fillna(0).to_numpy(dtype=bool)

                # Extract features: (M * HORIZON, F)
                X = panel.extract_features(sp, fl, el, orig, cond)

                # Multi-model prediction
                lgb_preds = [b.predict(X, num_threads=THREADS) for b in boosters]
                cb_pred = cb.predict_proba(X, thread_count=THREADS)[:, 1]
                p_ens = 0.75 * np.mean(lgb_preds, axis=0) + 0.25 * cb_pred

                # Reshape to (HORIZON, M)
                M = len(panel.links)
                p_mat = p_ens.reshape(HORIZON, M)

                for h_idx in range(HORIZON):
                    t_target = target_times[h_idx]
                    p_h = p_mat[h_idx]

                    # Decision logic
                    if cond == "queue_onset":
                        # In onset windows, congestion only breaks down at h=6 (T+30)
                        if h_idx == HORIZON - 1:
                            q_pred = (p_h >= tau).astype(int)
                            if q_pred.sum() < onset_fallback_k:
                                top_idx = np.argpartition(p_h, -onset_fallback_k)[-onset_fallback_k:]
                                q_pred[top_idx] = 1
                        else:
                            q_pred = np.zeros(M, dtype=int)
                    else:
                        # Ongoing windows
                        q_pred = (p_h >= tau).astype(int)

                    for m_idx in range(M):
                        records.append({
                            "window_id": wid,
                            "timestamp": t_target.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "link_id": panel.links[m_idx],
                            "panel": p_name,
                            "condition": cond,
                            "h": h_idx + 1,
                            "prob": float(p_h[m_idx]),
                            "queue_pred": int(q_pred[m_idx]),
                        })

    pred_df = pd.DataFrame(records)
    print(f"Generated raw predictions: {len(pred_df):,} rows in {time.time()-t0:.1f}s")
    print(f"Raw positive queue count: {(pred_df['queue_pred'] == 1).sum():,}")
    return pred_df


def apply_topological_shockwave_filter(pred_df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print(">>> APPLYING HIGH-CONFIDENCE TOPOLOGICAL SHOCKWAVE FILTER <<<")
    print("=" * 70)

    out_df = pred_df.copy()
    q_arr = out_df["queue_pred"].to_numpy().copy()
    p_arr = out_df["prob"].to_numpy()

    panels = out_df["panel"].unique()
    filled_1hole = 0
    filled_2hole = 0
    cleared_spikes = 0

    for p in panels:
        topo_path = DATA / "corridors" / p / "network" / "lwr_mainline_topology.csv"
        if not topo_path.exists():
            continue
        topo = pd.read_csv(topo_path, dtype=str)
        link2idx = {l: i for i, l in enumerate(topo["link_id"].tolist())}

        p_mask = (out_df["panel"] == p) & (out_df["condition"] == "queue_ongoing")
        p_indices = out_df[p_mask].index.to_numpy()
        sub = out_df.loc[p_indices].copy()
        sub["topo_idx"] = sub["link_id"].map(link2idx)
        sub = sub.dropna(subset=["topo_idx"]).sort_values(["window_id", "h", "topo_idx"])

        for (wid, h), grp in sub.groupby(["window_id", "h"]):
            idxs = grp.index.to_numpy()
            q_vals = q_arr[idxs]
            p_vals = p_arr[idxs]

            # 1. 1-link hole closure with probability gate (p >= 0.40)
            for i in range(1, len(q_vals) - 1):
                if q_vals[i-1] == 1 and q_vals[i] == 0 and q_vals[i+1] == 1:
                    if p_vals[i] >= 0.40:
                        q_arr[idxs[i]] = 1
                        filled_1hole += 1

            # 2. 2-link hole closure with probability gate (min(p1, p2) >= 0.40)
            for i in range(1, len(q_vals) - 2):
                if q_vals[i-1] == 1 and q_vals[i] == 0 and q_vals[i+1] == 0 and q_vals[i+2] == 1:
                    if p_vals[i] >= 0.40 and p_vals[i+1] >= 0.40:
                        q_arr[idxs[i]] = 1
                        q_arr[idxs[i+1]] = 1
                        filled_2hole += 2

            # 3. Isolated noise spike suppression (p <= 0.15)
            for i in range(1, len(q_vals) - 1):
                if q_vals[i-1] == 0 and q_vals[i] == 1 and q_vals[i+1] == 0:
                    if p_vals[i] <= 0.15:
                        q_arr[idxs[i]] = 0
                        cleared_spikes += 1

    out_df["queue_pred"] = q_arr
    print(f"Filter Summary:")
    print(f"  + 1-link high-confidence shockwave bridges filled: {filled_1hole}")
    print(f"  + 2-link high-confidence shockwave bridges filled: {filled_2hole}")
    print(f"  - Isolated low-confidence noise spikes cleared:   {cleared_spikes}")
    print(f"Final positive queue count: {(out_df['queue_pred'] == 1).sum():,} / {len(out_df):,}")

    # Format output columns matching submission contract
    clean_out = out_df[["window_id", "timestamp", "link_id", "queue_pred"]].copy()
    clean_out.to_csv(OUT_CSV, index=False)
    print(f"Saved Next-Gen Queue candidate to {OUT_CSV}")
    return clean_out


def main():
    boosters, cb = load_models()
    raw_df = predict_test_windows(boosters, cb, tau=0.350, onset_fallback_k=2)
    final_df = apply_topological_shockwave_filter(raw_df)


if __name__ == "__main__":
    main()
