"""Generate Task 2 queue predictions calibrated with empirical bottleneck priors.

Features:
  - Onset Windows:
    * Predictions strictly concentrated at horizon T+30 (h=6).
    * Dual 50/50 LightGBM + CatBoost ensemble.
    * Base threshold 0.55.
    * Bottleneck-calibrated fallback: when no link exceeds 0.55, selects the link
      maximizing S = p_model * (1 + 0.30 * p_EB), grounding predictions in physical
      corridor breakdown locations.
  - Ongoing Windows:
    * LightGBM ongoing booster with validated optimal threshold 0.45.
Outputs:
  outputs/submissions/queue_bottleneck_calibrated.csv
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path("trafficflowbench").resolve()
DATA = ROOT / "data" / "kaggle_public"
ASSETS = ROOT / "kaggle_research" / "anubis_assets"
PRIORS_DIR = ROOT / "outputs" / "priors"
OUT_DIR = ROOT / "outputs" / "submissions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["TFB_DATA_ROOT"] = str(DATA)
os.environ["TFB_OUTPUT_ROOT"] = str(OUT_DIR)
os.environ["TFB_THREADS"] = "6"

sys.path.insert(0, str(ASSETS))
import queue_ongoing as q_ongoing_module
import queue_features as q_onset_module
from queue_features import panel_names, HISTORY, HORIZON


def load_bottleneck_priors() -> dict[str, dict]:
    priors = {}
    for p in PRIORS_DIR.glob("*_bottleneck_prior.json"):
        panel = p.name.replace("_bottleneck_prior.json", "")
        priors[panel] = json.loads(p.read_text(encoding="utf-8"))
    return priors


def get_slot(ts: pd.Timestamp) -> int:
    return int(ts.weekday() * 288 + ts.hour * 12 + ts.minute // 5)


def main():
    t0 = time.time()
    print("=" * 70)
    print(">>> GENERATING BOTTLENECK-CALIBRATED TASK 2 QUEUE PREDICTIONS <<<")
    print("=" * 70)

    print("Loading Task 2 models...")
    m_on_lgb = lgb.Booster(model_file=str(ASSETS / "onset_lgb.txt"))
    m_ong_lgb = lgb.Booster(model_file=str(ASSETS / "ongoing_lgb.txt"))
    m_on_cb = CatBoostClassifier()
    m_on_cb.load_model(str(ASSETS / "onset.cbm"))
    print("Models loaded successfully.")

    priors = load_bottleneck_priors()
    print(f"Loaded bottleneck priors for {len(priors)} corridors.")

    out_file = OUT_DIR / "queue_bottleneck_calibrated.csv"
    first = True
    total_rows = 0

    splits = ("validation", "private")
    panels = panel_names()

    for split in splits:
        for panel_no, name in enumerate(panels):
            t_p = time.time()
            panel_onset = q_onset_module.Panel(name, panel_no)
            panel_ongoing = q_ongoing_module.Panel(name, panel_no)
            prior = priors.get(name, {})
            link_priors = prior.get("links", {})
            link_overall = prior.get("link_overall_freq", {})

            directory = DATA / "task2" / name / split
            windows = pd.read_csv(directory / "window_index.csv", dtype={"window_id": str})
            history = pd.read_parquet(directory / "window_history.parquet")
            template = pd.read_csv(directory / "sample_submission_queue.csv", dtype={"window_id": str, "link_id": str})
            pieces = []

            for row in windows.itertuples(index=False):
                origin = pd.Timestamp(row.forecast_origin)
                times = pd.date_range(end=origin - pd.Timedelta(minutes=5), periods=HISTORY, freq="5min", tz="UTC")
                index = pd.Index(times.strftime("%Y-%m-%dT%H:%M:%SZ"))
                h = history[history.window_id.astype(str) == str(row.window_id)].copy()
                h["link_id"] = h.link_id.astype(str)
                cond = str(row.condition)

                if cond == "queue_onset":
                    sp = h.pivot(index="timestamp", columns="link_id", values="speed_kmh").reindex(index=index, columns=panel_onset.links).to_numpy(dtype=np.float32)
                    fl = h.pivot(index="timestamp", columns="link_id", values="flow_vph").reindex(index=index, columns=panel_onset.links).to_numpy(dtype=np.float32)
                    observed = h.pivot(index="timestamp", columns="link_id", values="is_score_eligible").reindex(index=index, columns=panel_onset.links).fillna(0).to_numpy(dtype=bool)

                    X = panel_onset.features(sp, fl, observed, origin, cond)
                    p_lgb = m_on_lgb.predict(X, num_threads=6)
                    p_cb = m_on_cb.predict_proba(X, thread_count=6)[:, 1]
                    probs = 0.5 * p_lgb + 0.5 * p_cb

                    # The onset model predicts for the horizon T+30 step (-len(links):)
                    step_probs = probs[-len(panel_onset.links):].copy()
                    step_pred = step_probs >= 0.55

                    if not step_pred.any():
                        # Calibrated bottleneck fallback
                        slot_t30 = get_slot(origin + pd.Timedelta(minutes=30))
                        adj_scores = np.zeros(len(panel_onset.links), dtype=np.float64)
                        for li, lid in enumerate(panel_onset.links):
                            str_lid = str(lid)
                            # Lookup p_EB at slot_t30, or overall link frequency fallback
                            p_eb = link_priors.get(str_lid, {}).get(str(slot_t30), link_overall.get(str_lid, 0.05))
                            adj_scores[li] = step_probs[li] * (1.0 + 0.30 * p_eb)

                        # Pick the bottleneck link with maximum adjusted evidence
                        step_pred[np.argmax(adj_scores)] = True

                    pred = np.zeros(HORIZON * len(panel_onset.links), dtype=bool)
                    pred[-len(panel_onset.links):] = step_pred
                    links_order = panel_onset.links
                else:  # queue_ongoing
                    sp = h.pivot(index="timestamp", columns="link_id", values="speed_kmh").reindex(index=index, columns=panel_ongoing.links).to_numpy(dtype=np.float32)
                    fl = h.pivot(index="timestamp", columns="link_id", values="flow_vph").reindex(index=index, columns=panel_ongoing.links).to_numpy(dtype=np.float32)
                    observed = h.pivot(index="timestamp", columns="link_id", values="is_score_eligible").reindex(index=index, columns=panel_ongoing.links).fillna(0).to_numpy(dtype=bool)

                    X = panel_ongoing.features(sp, fl, observed, origin, cond)
                    probs = m_ong_lgb.predict(X, num_threads=6)
                    pred = probs >= 0.45
                    links_order = panel_ongoing.links

                timestamps = [(origin + pd.Timedelta(minutes=5 * step)).strftime("%Y-%m-%dT%H:%M:%SZ")
                              for step in range(1, HORIZON + 1)]
                pieces.append(pd.DataFrame({
                    "window_id": str(row.window_id),
                    "timestamp": np.repeat(timestamps, len(links_order)),
                    "link_id": np.tile(links_order, HORIZON),
                    "queue_pred": pred.astype(np.uint8),
                }))

            prediction = pd.concat(pieces, ignore_index=True)
            joined = template[["window_id", "timestamp", "link_id"]].merge(
                prediction, on=["window_id", "timestamp", "link_id"], how="left", validate="one_to_one"
            )
            assert not joined.queue_pred.isna().any(), f"NaN in queue predictions: {name} {split}"
            assert len(joined) == len(template), f"Row count mismatch in template: {name} {split}"

            joined.queue_pred = joined.queue_pred.astype(np.uint8)
            joined.to_csv(out_file, mode="w" if first else "a", header=first, index=False)
            first = False
            total_rows += len(joined)
            print(f"  {name} {split}: {len(joined):,} rows in {time.time()-t_p:.1f}s, pos={joined.queue_pred.sum():,}")

    print("-" * 70)
    print(f"Successfully generated {total_rows:,} queue predictions in {time.time()-t0:.1f}s -> {out_file.name}")
    assert total_rows == 174000, f"Expected 174,000 rows, got {total_rows}"
    print("ALL TASK 2 INTEGRITY CHECKS PASSED 100%!")


if __name__ == "__main__":
    main()
