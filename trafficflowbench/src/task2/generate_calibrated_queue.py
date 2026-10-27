"""Generate precisely calibrated Task 2 queue predictions using Anubis top-50 dual ensemble:
  - Onset: 83-feature 50/50 LightGBM + CatBoost blend with threshold 0.55 and guaranteed horizon peak.
  - Ongoing: 36-feature LightGBM with threshold 0.40.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("trafficflowbench").resolve()
ASSETS = ROOT / "kaggle_research" / "anubis_assets"
DATA = ROOT / "data" / "kaggle_public"
OUT = ROOT / "outputs" / "submissions"

os.environ["TFB_DATA_ROOT"] = str(DATA)
os.environ["TFB_OUTPUT_ROOT"] = str(OUT)
os.environ["TFB_THREADS"] = "4"

sys.path.insert(0, str(ASSETS))
import queue_ongoing as q_ongoing_module
import queue_features as q_onset_module
from queue_features import panel_names, HISTORY, HORIZON
import lightgbm as lgb
from catboost import CatBoostClassifier


def main():
    print("Loading Task 2 queue models...")
    m_on_lgb = lgb.Booster(model_file=str(ASSETS / "onset_lgb.txt"))
    m_ong_lgb = lgb.Booster(model_file=str(ASSETS / "ongoing_lgb.txt"))
    m_on_cb = CatBoostClassifier()
    m_on_cb.load_model(str(ASSETS / "onset.cbm"))
    print("All 3 models loaded successfully.")

    out_file = OUT / "queue_calibrated.csv"
    first = True
    total_rows = 0

    splits = ("validation", "private")
    panels = panel_names()
    print(f"Scoring corridors: {panels} across {splits}...")

    for split in splits:
        for panel_no, name in enumerate(panels):
            panel_onset = q_onset_module.Panel(name, panel_no)
            panel_ongoing = q_ongoing_module.Panel(name, panel_no)

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
                    p_lgb = m_on_lgb.predict(X, num_threads=4)
                    p_cb = m_on_cb.predict_proba(X, thread_count=4)[:, 1]
                    probs = 0.5 * p_lgb + 0.5 * p_cb

                    # The onset model predicts for the last horizon step (-len(links):)
                    pred = np.zeros(HORIZON * len(panel_onset.links), dtype=bool)
                    step_probs = probs[-len(panel_onset.links):]
                    step_pred = step_probs >= 0.55
                    if not step_pred.any():
                        step_pred[np.argmax(step_probs)] = True
                    pred[-len(panel_onset.links):] = step_pred
                    links_order = panel_onset.links
                else:  # queue_ongoing
                    sp = h.pivot(index="timestamp", columns="link_id", values="speed_kmh").reindex(index=index, columns=panel_ongoing.links).to_numpy(dtype=np.float32)
                    fl = h.pivot(index="timestamp", columns="link_id", values="flow_vph").reindex(index=index, columns=panel_ongoing.links).to_numpy(dtype=np.float32)
                    observed = h.pivot(index="timestamp", columns="link_id", values="is_score_eligible").reindex(index=index, columns=panel_ongoing.links).fillna(0).to_numpy(dtype=bool)

                    X = panel_ongoing.features(sp, fl, observed, origin, cond)
                    probs = m_ong_lgb.predict(X, num_threads=4)
                    pred = probs >= 0.40
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
            print(f"Predicted {name} {split}: {len(joined)} rows, pos share: {joined.queue_pred.mean():.4f}")

    print(f"Done! Successfully wrote {total_rows:,} rows to {out_file}")


if __name__ == "__main__":
    main()
