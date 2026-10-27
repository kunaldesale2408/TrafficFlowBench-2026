"""Tune Onset Fallback Strategy on 211 validation onset windows.
Evaluates:
  - Base threshold: [0.55, 0.52, 0.50, 0.48, 0.45]
  - Top-k bottleneck links: [1, 2, 3, 4, 5, 6, 8, 10]
  - Probability-calibrated relative thresholding
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

os.environ["TFB_DATA_ROOT"] = str(DATA)
os.environ["TFB_OUTPUT_ROOT"] = str(ROOT / "scratch")
os.environ["TFB_THREADS"] = "6"

sys.path.insert(0, str(ASSETS))
import queue_features as q_module
from queue_features import panel_names, Panel

def load_bottleneck_priors():
    priors = {}
    for p in PRIORS_DIR.glob("*_bottleneck_prior.json"):
        priors[p.name.replace("_bottleneck_prior.json", "")] = json.loads(p.read_text(encoding="utf-8"))
    return priors

def main():
    t0 = time.time()
    print("=" * 65)
    print(">>> TUNING ONSET QUEUE EVALUATION STRATEGY <<<")
    print("=" * 65)

    panels = panel_names()
    panel_objs = {name: Panel(name, idx) for idx, name in enumerate(panels)}
    priors = load_bottleneck_priors()

    print("Collecting validation windows...")
    X, y, weights, valid_windows, feature_count = q_module.collect()
    onset_val = [w for w in valid_windows if w.condition == "queue_onset"]
    print(f"Total onset validation windows: {len(onset_val)}")

    print("Loading models...")
    m_on_lgb = lgb.Booster(model_file=str(ASSETS / "onset_lgb.txt"))
    m_on_cb = CatBoostClassifier()
    m_on_cb.load_model(str(ASSETS / "onset.cbm"))

    # Precompute model predictions for all onset windows
    raw_preds = []
    for w in onset_val:
        p_l = m_on_lgb.predict(w.X, num_threads=6)
        p_c = m_on_cb.predict_proba(w.X, thread_count=6)[:, 1]
        p_avg = 0.5 * p_l + 0.5 * p_c
        n_links = len(panel_objs[w.panel].links)
        raw_preds.append((w, p_avg[-n_links:], panel_objs[w.panel]))
    print(f"Inference complete on {len(raw_preds)} onset windows in {time.time()-t0:.1f}s")

    # Evaluate grid
    print("\n" + "-" * 75)
    print(f"{'Threshold':<10} | {'Top-k':<8} | {'Onset IoU':<12} | {'Empty Window Fallbacks':<25}")
    print("-" * 75)

    best_iou = 0.0
    best_cfg = None

    for thresh in [0.55, 0.52, 0.50, 0.48, 0.45]:
        for k in [1, 2, 3, 4, 5, 6, 8, 10]:
            ious = []
            fallbacks = 0
            for w, step_probs, p_obj in raw_preds:
                step_pred = step_probs >= thresh
                if not step_pred.any():
                    fallbacks += 1
                    # Bottleneck prior fallback
                    prior = priors.get(w.panel, {})
                    link_overall = prior.get("link_overall_freq", {})
                    adj = np.zeros(len(p_obj.links), dtype=np.float64)
                    for li, lid in enumerate(p_obj.links):
                        p_eb = link_overall.get(str(lid), 0.05)
                        adj[li] = step_probs[li] * (1.0 + 0.30 * p_eb)
                    
                    top_indices = np.argsort(adj)[-k:]
                    step_pred[top_indices] = True

                pred = np.zeros(6 * len(p_obj.links), dtype=bool)
                pred[-len(p_obj.links):] = step_pred
                true = w.y[w.eligible].astype(bool)
                p_eval = pred[w.eligible]
                inter = (p_eval & true).sum()
                union = (p_eval | true).sum()
                ious.append(inter / union if union > 0 else 1.0)
            
            mean_iou = float(np.mean(ious))
            if mean_iou > best_iou:
                best_iou = mean_iou
                best_cfg = (thresh, k)
                print(f"{thresh:<10.2f} | {k:<8d} | {mean_iou:<12.5f} | {fallbacks}/{len(raw_preds)} *** NEW BEST ***")
            elif k in (1, 2, 3, 5):
                print(f"{thresh:<10.2f} | {k:<8d} | {mean_iou:<12.5f} | {fallbacks}/{len(raw_preds)}")

    print("-" * 75)
    print(f"\nGLOBAL OPTIMAL ONSET CONFIGURATION: Threshold = {best_cfg[0]}, Fallback top_k = {best_cfg[1]} -> IoU: {best_iou:.5f}")
    baseline_iou = [float(np.mean([(pred[-len(p_obj.links):] >= 0.55)[w.eligible[-len(p_obj.links):]].sum() / max(1, w.y[w.eligible].sum()) for w, pred, p_obj in raw_preds]))]
    print("=" * 65)

if __name__ == "__main__":
    main()
