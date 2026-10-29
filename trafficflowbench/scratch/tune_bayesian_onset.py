"""Evaluate and tune Bayesian Prior-Weighted Onset Queue Forecasting.
Tests:
  1. Base threshold vs Prior-adjusted threshold: tau_i = tau_base - gamma * p_eb
  2. Maximum cluster pruning: limit onset predictions per window to top-k (k=1 or 2)
  3. Dynamic probability scoring: score_i = prob_i * (1 + beta * p_eb)
Evaluates exact IoU on all validation onset windows.
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
from queue_features import panel_names, Panel, HISTORY, HORIZON


def load_bottleneck_priors() -> dict[str, dict]:
    priors = {}
    for p in PRIORS_DIR.glob("*_bottleneck_prior.json"):
        priors[p.name.replace("_bottleneck_prior.json", "")] = json.loads(p.read_text(encoding="utf-8"))
    return priors


def get_slot(ts: pd.Timestamp) -> int:
    return int(ts.weekday() * 288 + ts.hour * 12 + ts.minute // 5)


def main():
    t0 = time.time()
    print("=" * 75)
    print(">>> TUNING BAYESIAN PRIOR-WEIGHTED ONSET QUEUE FORECASTING <<<")
    print("=" * 75)

    panels = panel_names()
    panel_objs = {name: Panel(name, idx) for idx, name in enumerate(panels)}
    priors = load_bottleneck_priors()

    cache_file = ROOT / "scratch" / "onset_val_dataset.pkl"
    if cache_file.exists():
        print(f"Loading cached dataset from {cache_file}...")
        import pickle
        with open(cache_file, "rb") as f:
            dataset = pickle.load(f)
        print(f"Loaded {len(dataset)} onset windows from cache.")
    else:
        print("Collecting validation onset windows...")
        _, _, _, valid_windows, _ = q_module.collect()
        onset_val = [w for w in valid_windows if w.condition == "queue_onset"]
        print(f"Total onset validation windows: {len(onset_val)}")

        print("Loading Task 2 onset models...")
        m_on_lgb = lgb.Booster(model_file=str(ASSETS / "onset_lgb.txt"))
        m_on_cb = CatBoostClassifier()
        m_on_cb.load_model(str(ASSETS / "onset.cbm"))

        # Precompute raw model probabilities and prior values for all onset validation windows
        dataset = []
        for w in onset_val:
            p_l = m_on_lgb.predict(w.X, num_threads=6)
            p_c = m_on_cb.predict_proba(w.X, thread_count=6)[:, 1]
            probs = 0.5 * p_l + 0.5 * p_c

            p_obj = panel_objs[w.panel]
            n_links = len(p_obj.links)
            step_probs = probs[-n_links:].copy()

            # Get slot for T+30 from feature columns 6 (hour) and 9 (dow)
            hour = float(w.X[0, 6])
            dow = int(round(float(w.X[0, 9])))
            t30_hour = hour + 0.5
            t30_dow = dow
            if t30_hour >= 24.0:
                t30_hour -= 24.0
                t30_dow = (t30_dow + 1) % 7
            slot_t30 = int(t30_dow * 288 + round(t30_hour * 12))
            slot_t30 = max(0, min(2015, slot_t30))
            prior_dict = priors.get(w.panel, {})
            link_priors = prior_dict.get("links", {})
            link_overall = prior_dict.get("link_overall_freq", {})

            p_eb = np.zeros(n_links, dtype=np.float64)
            for li, lid in enumerate(p_obj.links):
                str_lid = str(lid)
                p_eb[li] = link_priors.get(str_lid, {}).get(str(slot_t30), link_overall.get(str_lid, 0.02))

            # Ground truth at horizon step 6 (T+30)
            true_full = w.y[w.eligible].astype(bool)
            true_t30 = w.y[-n_links:].astype(bool)
            elig_t30 = w.eligible[-n_links:].astype(bool)

            dataset.append({
                "panel": w.panel,
                "n_links": n_links,
                "step_probs": step_probs,
                "p_eb": p_eb,
                "true_full": true_full,
                "true_t30": true_t30,
                "elig_t30": elig_t30,
                "eligible": w.eligible,
            })
        print(f"Precomputed features on {len(dataset)} onset windows in {time.time()-t0:.1f}s")
        import pickle
        with open(cache_file, "wb") as f:
            pickle.dump(dataset, f)
        print(f"Saved cache to {cache_file}")

    # Baseline evaluation: current S23/S24 onset logic (thresh=0.55, top-1 fallback)
    base_ious = []
    for d in dataset:
        probs = d["step_probs"].copy()
        pred_step = probs >= 0.55
        if not pred_step.any():
            adj = probs * (1.0 + 0.30 * d["p_eb"])
            pred_step[np.argmax(adj)] = True
        
        pred_full = np.zeros(6 * d["n_links"], dtype=bool)
        pred_full[-d["n_links"]:] = pred_step
        p_eval = pred_full[d["eligible"]]
        inter = (p_eval & d["true_full"]).sum()
        union = (p_eval | d["true_full"]).sum()
        base_ious.append(inter / union if union > 0 else 1.0)
    baseline_mean_iou = float(np.mean(base_ious))
    print(f"\nBaseline S23/S24 Onset Validation IoU: {baseline_mean_iou:.5f}")

    # Comprehensive Grid Search
    results = []

    # Strategy 1: Score = prob * (1 + beta * eb) >= tau (Adaptive or Fixed tau)
    for beta in [0.0, 0.15, 0.30, 0.50, 0.80, 1.20]:
        for thresh in [0.45, 0.50, 0.55, 0.58, 0.60, 0.65, 0.70]:
            for max_k in [1, 2, 3]:
                for gamma in [0.0, 0.10, 0.20]:
                    ious = []
                    for d in dataset:
                        p_raw = d["step_probs"]
                        eb = d["p_eb"]
                        score = p_raw * (1.0 + beta * eb)
                        tau = thresh - gamma * eb
                        cand = score >= tau

                        if not cand.any():
                            pred_step = np.zeros(len(p_raw), dtype=bool)
                            pred_step[np.argmax(score)] = True
                        else:
                            cand_idx = np.where(cand)[0]
                            if len(cand_idx) > max_k:
                                sorted_cand = cand_idx[np.argsort(score[cand_idx])[-max_k:]]
                                pred_step = np.zeros(len(p_raw), dtype=bool)
                                pred_step[sorted_cand] = True
                            else:
                                pred_step = cand.copy()

                        pred_full = np.zeros(6 * d["n_links"], dtype=bool)
                        pred_full[-d["n_links"]:] = pred_step
                        p_eval = pred_full[d["eligible"]]
                        inter = (p_eval & d["true_full"]).sum()
                        union = (p_eval | d["true_full"]).sum()
                        ious.append(inter / union if union > 0 else 1.0)

                    m_iou = float(np.mean(ious))
                    results.append({
                        "mode": "bayes_score",
                        "beta": beta,
                        "thresh": thresh,
                        "max_k": max_k,
                        "gamma": gamma,
                        "iou": m_iou,
                    })

    # Strategy 2: Raw prob >= tau (no false positives), fallback = argmax(prob * (1 + beta * eb))
    for beta in [0.0, 0.15, 0.30, 0.50, 0.80, 1.20, 2.0]:
        for thresh in [0.45, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70]:
            for max_k in [1, 2, 3]:
                ious = []
                for d in dataset:
                    p_raw = d["step_probs"]
                    eb = d["p_eb"]
                    cand = p_raw >= thresh

                    if not cand.any():
                        pred_step = np.zeros(len(p_raw), dtype=bool)
                        score = p_raw * (1.0 + beta * eb)
                        pred_step[np.argmax(score)] = True
                    else:
                        cand_idx = np.where(cand)[0]
                        if len(cand_idx) > max_k:
                            sorted_cand = cand_idx[np.argsort(p_raw[cand_idx])[-max_k:]]
                            pred_step = np.zeros(len(p_raw), dtype=bool)
                            pred_step[sorted_cand] = True
                        else:
                            pred_step = cand.copy()

                    pred_full = np.zeros(6 * d["n_links"], dtype=bool)
                    pred_full[-d["n_links"]:] = pred_step
                    p_eval = pred_full[d["eligible"]]
                    inter = (p_eval & d["true_full"]).sum()
                    union = (p_eval | d["true_full"]).sum()
                    ious.append(inter / union if union > 0 else 1.0)

                m_iou = float(np.mean(ious))
                results.append({
                    "mode": "raw_thresh_fallback_eb",
                    "beta": beta,
                    "thresh": thresh,
                    "max_k": max_k,
                    "gamma": 0.0,
                    "iou": m_iou,
                })

    # Sort results by validation IoU descending
    results.sort(key=lambda x: x["iou"], reverse=True)

    print("\n" + "=" * 90)
    print(f"TOP 15 ONSET FORECASTING CONFIGURATIONS (Baseline = {baseline_mean_iou:.5f}):")
    print("=" * 90)
    print(f"{'Rank':<5} | {'Mode':<24} | {'Beta':<6} | {'Thresh':<7} | {'Max-K':<6} | {'Gamma':<6} | {'IoU':<10} | {'Delta':<10}")
    print("-" * 90)
    for r, res in enumerate(results[:15], 1):
        delta = res['iou'] - baseline_mean_iou
        print(f"{r:<5} | {res['mode']:<24} | {res['beta']:<6.2f} | {res['thresh']:<7.2f} | {res['max_k']:<6d} | {res['gamma']:<6.2f} | {res['iou']:<10.5f} | {delta:<+10.5f}")
    print("=" * 90)

    best = results[0]
    best_iou = best["iou"]
    print(f"\nOPTIMAL ONSET CONFIGURATION:")
    print(f"  - Mode:                             {best['mode']}")
    print(f"  - Beta (Prior boost factor):        {best['beta']}")
    print(f"  - Base Threshold:                   {best['thresh']}")
    print(f"  - Max Cluster Size (Max-K links):   {best['max_k']}")
    print(f"  - Gamma (Prior threshold relax):    {best['gamma']}")
    print(f"  - Final Validation Onset IoU:       {best_iou:.5f} (Gain: {best_iou - baseline_mean_iou:+.5f})")
    print("=" * 75)

    # Save configuration to outputs/models
    cfg_data = {
        "mode": best["mode"],
        "beta": best["beta"],
        "thresh": best["thresh"],
        "max_k": best["max_k"],
        "gamma": best["gamma"],
        "baseline_iou": baseline_mean_iou,
        "best_iou": best_iou,
        "gain": best_iou - baseline_mean_iou,
    }
    out_cfg = ROOT / "outputs" / "models" / "bayesian_onset_config.json"
    with open(out_cfg, "w") as f:
        json.dump(cfg_data, f, indent=2)
    print(f"Saved optimal Bayesian onset config -> {out_cfg.name}")


if __name__ == "__main__":
    main()
