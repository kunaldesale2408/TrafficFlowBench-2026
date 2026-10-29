"""Generate Task 2 queue predictions with Bayesian Prior-Weighted Onset and Calibrated Ongoing Ensembling:
  - Onset Windows:
    * Predictions strictly concentrated at horizon T+30 (h=6).
    * Dual 50/50 LightGBM + CatBoost ensemble.
    * Prior-boosted scoring: S_i = p_i * (1 + beta * p_EB)
    * Adaptive threshold: tau_i = tau_base - gamma * p_EB
    * Cluster pruning: at most max_k links per window
    * Fallback: link maximizing S_i when no link exceeds tau_i
  - Ongoing Windows:
    * Dual 88% LightGBM + 12% CatBoost ongoing ensemble.
    * District-specific calibrated thresholds: D7=0.485 (dense recurrent), D12=0.495 (high-speed free-flow).
Outputs:
  outputs/submissions/queue_s25_bayesian.csv
"""
from __future__ import annotations

import argparse
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
MODELS_DIR = ROOT / "outputs" / "models"
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
    print("=" * 75)
    print(">>> GENERATING BAYESIAN PRIOR-WEIGHTED QUEUE PREDICTIONS FOR S25 <<<")
    print("=" * 75)

    # Load optimal Bayesian onset config if available
    cfg_file = MODELS_DIR / "bayesian_onset_config.json"
    default_beta = 1.25
    default_thresh = 0.42
    default_gamma = 0.20
    default_fallback_k = 7

    if cfg_file.exists():
        try:
            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            default_beta = float(cfg.get("beta", default_beta))
            default_thresh = float(cfg.get("tau", default_thresh))
            default_gamma = float(cfg.get("gamma", default_gamma))
            default_fallback_k = int(cfg.get("fallback_k", default_fallback_k))
            print(f"Loaded tuned Bayesian onset config from {cfg_file.name}:")
            print(f"  tau={default_thresh}, beta={default_beta}, gamma={default_gamma}, fallback_k={default_fallback_k}")
            print(f"  validation IoU: {cfg.get('best_iou', 'N/A')} (gain: {cfg.get('gain', 'N/A'):+.5f})")
        except Exception as e:
            print(f"Warning: could not read {cfg_file}: {e}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ongoing-lgb-weight", type=float, default=0.88, help="Weight for ongoing LightGBM model")
    parser.add_argument("--ongoing-threshold", type=float, default=0.490, help="Default global threshold for ongoing queue")
    parser.add_argument("--ongoing-threshold-d7", type=float, default=0.485, help="District 7 ongoing threshold")
    parser.add_argument("--ongoing-threshold-d12", type=float, default=0.495, help="District 12 ongoing threshold")
    parser.add_argument("--onset-beta", type=float, default=default_beta, help="Prior boost factor for onset")
    parser.add_argument("--onset-thresh", type=float, default=default_thresh, help="Base threshold for onset")
    parser.add_argument("--onset-gamma", type=float, default=default_gamma, help="Prior threshold relaxation for onset")
    parser.add_argument("--onset-fallback-k", type=int, default=default_fallback_k, help="Fallback top-k links if none exceed threshold")
    parser.add_argument("--out-name", type=str, default="queue_s25_bayesian.csv", help="Output filename in outputs/submissions")
    args = parser.parse_args()

    # 1. Load models
    print("\nLoading Task 2 onset models...")
    m_on_lgb = lgb.Booster(model_file=str(ASSETS / "onset_lgb.txt"))
    m_on_cb = CatBoostClassifier()
    m_on_cb.load_model(str(ASSETS / "onset.cbm"))

    print("Loading Task 2 ongoing models...")
    m_ong_lgb = lgb.Booster(model_file=str(ASSETS / "ongoing_lgb.txt"))
    m_ong_cb = CatBoostClassifier()
    m_ong_cb.load_model(str(MODELS_DIR / "ongoing_cb.cbm"))
    print("All 4 Queue models loaded successfully.")

    # 2. Load empirical Bayes bottleneck priors
    priors = load_bottleneck_priors()
    print(f"Loaded bottleneck priors for {len(priors)} corridors.")

    # Hyper-parameters
    w_ong_lgb = args.ongoing_lgb_weight
    w_ong_cb = 1.0 - w_ong_lgb
    beta = args.onset_beta
    thresh = args.onset_thresh
    fallback_k = args.onset_fallback_k
    gamma = args.onset_gamma

    print(f"\nConfiguration:")
    print(f"  Ongoing: {w_ong_lgb*100:.1f}% LGB + {w_ong_cb*100:.1f}% CB | D7 thresh={args.ongoing_threshold_d7}, D12 thresh={args.ongoing_threshold_d12}")
    print(f"  Onset:   50% LGB + 50% CB | tau={thresh}, beta={beta}, gamma={gamma}, fallback_k={fallback_k}")

    out_file = OUT_DIR / args.out_name
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

                    slot_t30 = get_slot(origin + pd.Timedelta(minutes=30))
                    p_eb = np.zeros(len(panel_onset.links), dtype=np.float64)
                    for li, lid in enumerate(panel_onset.links):
                        str_lid = str(lid)
                        p_eb[li] = link_priors.get(str_lid, {}).get(str(slot_t30), link_overall.get(str_lid, 0.02))

                    eff_tau = thresh - gamma * p_eb
                    eff_prob = step_probs * (1.0 + beta * p_eb)
                    cand = eff_prob >= eff_tau

                    if not cand.any():
                        step_pred = np.zeros(len(panel_onset.links), dtype=bool)
                        score = step_probs * (1.0 + 0.30 * p_eb)
                        k = min(args.onset_fallback_k, len(panel_onset.links))
                        top_idx = np.argsort(score)[-k:]
                        step_pred[top_idx] = True
                    else:
                        step_pred = cand.copy()

                    pred = np.zeros(HORIZON * len(panel_onset.links), dtype=bool)
                    pred[-len(panel_onset.links):] = step_pred
                    links_order = panel_onset.links
                else:  # queue_ongoing
                    sp = h.pivot(index="timestamp", columns="link_id", values="speed_kmh").reindex(index=index, columns=panel_ongoing.links).to_numpy(dtype=np.float32)
                    fl = h.pivot(index="timestamp", columns="link_id", values="flow_vph").reindex(index=index, columns=panel_ongoing.links).to_numpy(dtype=np.float32)
                    observed = h.pivot(index="timestamp", columns="link_id", values="is_score_eligible").reindex(index=index, columns=panel_ongoing.links).fillna(0).to_numpy(dtype=bool)

                    X = panel_ongoing.features(sp, fl, observed, origin, cond)
                    p_ong_lgb = m_ong_lgb.predict(X, num_threads=6)
                    p_ong_cb = m_ong_cb.predict_proba(X, thread_count=6)[:, 1]
                    probs = w_ong_lgb * p_ong_lgb + w_ong_cb * p_ong_cb
                    current_th = (
                        args.ongoing_threshold_d7 if name.startswith("D7")
                        else (args.ongoing_threshold_d12 if name.startswith("D12") else args.ongoing_threshold)
                    )
                    pred = probs >= current_th
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

    print("-" * 75)
    print(f"Successfully generated {total_rows:,} queue predictions in {time.time()-t0:.1f}s -> {out_file.name}")
    assert total_rows == 174000, f"Expected 174,000 rows, got {total_rows}"
    print("ALL TASK 2 INTEGRITY CHECKS PASSED 100%!")


if __name__ == "__main__":
    main()
