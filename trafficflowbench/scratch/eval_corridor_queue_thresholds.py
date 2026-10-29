"""Evaluate Ongoing Queue Decision Thresholds Per Corridor and District.
Determines whether District-specific thresholds (D7 vs D12) or corridor-specific thresholds
outperform the global 0.490 threshold on validation IoU.
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
MODELS_DIR = ROOT / "outputs" / "models"

os.environ["TFB_DATA_ROOT"] = str(DATA)
os.environ["TFB_OUTPUT_ROOT"] = str(ROOT / "scratch")
os.environ["TFB_THREADS"] = "6"

sys.path.insert(0, str(ASSETS))
import queue_ongoing as q_ongoing_module
from queue_features import panel_names, HISTORY, HORIZON


def main():
    print("=" * 75)
    print(">>> EVALUATING ONGOING QUEUE THRESHOLDS PER CORRIDOR & DISTRICT <<<")
    print("=" * 75)

    print("Loading ongoing models...")
    m_ong_lgb = lgb.Booster(model_file=str(ASSETS / "ongoing_lgb.txt"))
    m_ong_cb = CatBoostClassifier()
    m_ong_cb.load_model(str(MODELS_DIR / "ongoing_cb.cbm"))

    w_ong_lgb = 0.88
    w_ong_cb = 0.12

    truth_all = pd.read_parquet(ROOT / "scratch" / "train_queue_truth.parquet")
    truth_all["window_id"] = truth_all.window_id.astype(str)
    truth_all["timestamp"] = pd.to_datetime(truth_all.timestamp, utc=True)
    truth_all["link_id"] = truth_all.link_id.astype(str)
    print(f"Loaded train truth labels: {len(truth_all):,} rows.")

    panels = panel_names()
    corridor_data = {}

    for panel_no, name in enumerate(panels):
        t0 = time.time()
        panel_ongoing = q_ongoing_module.Panel(name, panel_no)
        folder = DATA / "task2" / name / "train"
        if not folder.exists():
            continue

        windows = pd.read_csv(folder / "window_index.csv", dtype={"window_id": str})
        history = pd.read_parquet(folder / "window_history.parquet")

        ongoing_windows = windows[windows.condition == "queue_ongoing"]
        p_list, t_list = [], []

        for row in ongoing_windows.itertuples(index=False):
            origin = pd.Timestamp(row.forecast_origin)
            times = pd.date_range(end=origin - pd.Timedelta(minutes=5), periods=HISTORY, freq="5min", tz="UTC")
            index = pd.Index(times.strftime("%Y-%m-%dT%H:%M:%SZ"))
            h = history[history.window_id.astype(str) == str(row.window_id)].copy()
            h["link_id"] = h.link_id.astype(str)

            sp = h.pivot(index="timestamp", columns="link_id", values="speed_kmh").reindex(index=index, columns=panel_ongoing.links).to_numpy(dtype=np.float32)
            fl = h.pivot(index="timestamp", columns="link_id", values="flow_vph").reindex(index=index, columns=panel_ongoing.links).to_numpy(dtype=np.float32)
            observed = h.pivot(index="timestamp", columns="link_id", values="is_score_eligible").reindex(index=index, columns=panel_ongoing.links).fillna(0).to_numpy(dtype=bool)

            X = panel_ongoing.features(sp, fl, observed, origin, "queue_ongoing")
            p_lgb = m_ong_lgb.predict(X, num_threads=6)
            p_cb = m_ong_cb.predict_proba(X, thread_count=6)[:, 1]
            probs = w_ong_lgb * p_lgb + w_ong_cb * p_cb

            tr = truth_all[truth_all.window_id == str(row.window_id)]
            t_times = [pd.Timestamp(origin + pd.Timedelta(minutes=5 * s)) for s in range(1, HORIZON + 1)]
            
            # Match truth to (t_times, panel_ongoing.links)
            tr_pivot = tr.pivot(index="timestamp", columns="link_id", values="queue_true").reindex(index=t_times, columns=panel_ongoing.links).fillna(False).to_numpy(dtype=bool)
            el_pivot = tr.pivot(index="timestamp", columns="link_id", values="is_score_eligible").reindex(index=t_times, columns=panel_ongoing.links).fillna(False).to_numpy(dtype=bool)

            p_list.append(probs[el_pivot.ravel()])
            t_list.append(tr_pivot.ravel()[el_pivot.ravel()])

        all_p = np.concatenate(p_list)
        all_t = np.concatenate(t_list)
        corridor_data[name] = {"probs": all_p, "truth": all_t}
        pos_rate = float(np.mean(all_t))
        print(f"Loaded {name}: {len(all_p):,} cells (Queue prevalence = {pos_rate:.2%}) in {time.time()-t0:.1f}s")

    # Grid search thresholds per corridor
    threshold_range = np.arange(0.40, 0.56, 0.01)
    results = {}

    print("\n" + "=" * 75)
    print(f"{'Corridor':<15} | {'Best Thresh':<12} | {'Max IoU':<10} | {'IoU @ 0.49':<10} | {'IoU Gain':<10}")
    print("-" * 75)

    global_p, global_t = [], []
    d7_p, d7_t = [], []
    d12_p, d12_t = [], []

    for name, cdata in corridor_data.items():
        p = cdata["probs"]
        t = cdata["truth"]
        global_p.append(p)
        global_t.append(t)
        if name.startswith("D7"):
            d7_p.append(p)
            d7_t.append(t)
        elif name.startswith("D12"):
            d12_p.append(p)
            d12_t.append(t)

        best_iou, best_th = 0.0, 0.49
        iou_at_049 = 0.0

        for th in threshold_range:
            pred = p >= th
            inter = np.sum(pred & t)
            union = np.sum(pred | t)
            iou = inter / union if union > 0 else 0.0
            if abs(th - 0.49) < 1e-4:
                iou_at_049 = iou
            if iou > best_iou:
                best_iou = iou
                best_th = th

        gain = best_iou - iou_at_049
        results[name] = {"best_th": round(float(best_th), 3), "best_iou": round(float(best_iou), 4), "iou_at_049": round(float(iou_at_049), 4)}
        print(f"{name:<15} | {best_th:<12.2f} | {best_iou:<10.4f} | {iou_at_049:<10.4f} | {gain:+10.4f}")

    # Evaluate Global vs District
    print("-" * 75)
    all_p = np.concatenate(global_p)
    all_t = np.concatenate(global_t)
    best_g_iou, best_g_th = 0.0, 0.49
    for th in threshold_range:
        pred = all_p >= th
        iou = np.sum(pred & all_t) / np.sum(pred | all_t)
        if iou > best_g_iou:
            best_g_iou, best_g_th = iou, th
    iou_g_049 = np.sum((all_p >= 0.49) & all_t) / np.sum((all_p >= 0.49) | all_t)
    print(f"{'OVERALL GLOBAL':<15} | {best_g_th:<12.2f} | {best_g_iou:<10.4f} | {iou_g_049:<10.4f} | {best_g_iou - iou_g_049:+10.4f}")

    # District 7
    all_d7_p = np.concatenate(d7_p)
    all_d7_t = np.concatenate(d7_t)
    best_d7_iou, best_d7_th = 0.0, 0.49
    for th in threshold_range:
        pred = all_d7_p >= th
        iou = np.sum(pred & all_d7_t) / np.sum(pred | all_d7_t)
        if iou > best_d7_iou:
            best_d7_iou, best_d7_th = iou, th
    iou_d7_049 = np.sum((all_d7_p >= 0.49) & all_d7_t) / np.sum((all_d7_p >= 0.49) | all_d7_t)
    print(f"{'DISTRICT 7':<15} | {best_d7_th:<12.2f} | {best_d7_iou:<10.4f} | {iou_d7_049:<10.4f} | {best_d7_iou - iou_d7_049:+10.4f}")

    # District 12
    if len(d12_p) > 0:
        all_d12_p = np.concatenate(d12_p)
        all_d12_t = np.concatenate(d12_t)
        best_d12_iou, best_d12_th = 0.0, 0.49
        for th in threshold_range:
            pred = all_d12_p >= th
            iou = np.sum(pred & all_d12_t) / np.sum(pred | all_d12_t)
            if iou > best_d12_iou:
                best_d12_iou, best_d12_th = iou, th
        iou_d12_049 = np.sum((all_d12_p >= 0.49) & all_d12_t) / np.sum((all_d12_p >= 0.49) | all_d12_t)
        print(f"{'DISTRICT 12':<15} | {best_d12_th:<12.2f} | {best_d12_iou:<10.4f} | {iou_d12_049:<10.4f} | {best_d12_iou - iou_d12_049:+10.4f}")

    print("=" * 75)

    # Save threshold map
    thresh_cfg = {
        "global_th": float(best_g_th),
        "district_th": {
            "D7": float(best_d7_th),
            "D12": float(best_d12_th) if len(d12_p) > 0 else float(best_g_th),
        },
        "corridor_th": {k: v["best_th"] for k, v in results.items()},
    }
    out_cfg = ROOT / "outputs" / "models" / "calibrated_queue_thresholds.json"
    with open(out_cfg, "w") as f:
        json.dump(thresh_cfg, f, indent=2)
    print(f"Saved calibrated queue thresholds -> {out_cfg.name}")


if __name__ == "__main__":
    main()
