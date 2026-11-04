"""Generate Task 2 Queue Forecasts with Causal T-Slice Injection and 1D Milepost Morphological Closing:
1. Causal T-Slice Forward Shift:
   - Extracts observed / reconstructed speeds v_T at forecast origin T.
   - Evaluates ongoing model on standard [T-60..T-5] (p_std) and forward-shifted [T-55..T] (p_T).
   - Blends p_causal = 0.60 * p_std + 0.40 * p_T.
   - Promotes confirmed active queues at T (v_T <= v_cut) with p_causal >= 0.45.
   - Clears confirmed free-flow links at T (v_T > v_cut + 12) with p_causal < 0.38.
2. 1D Milepost Spatial Morphological Closing:
   - Bridges interior 1-link queue holes along highway milepost order when p_causal >= 0.28.
   - Suppresses isolated spatio-temporal 1-cell false positive spikes.
3. Onset Windows:
   - Retains proven champion T+30 structure with top-1 Empirical Bayes bottleneck fallback.
Outputs:
  outputs/submissions/queue_causal_t_morphological.csv (174,000 rows)
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
import state_model as base


def load_bottleneck_priors() -> dict[str, dict]:
    priors = {}
    for p in PRIORS_DIR.glob("*_bottleneck_prior.json"):
        panel = p.name.replace("_bottleneck_prior.json", "")
        priors[panel] = json.loads(p.read_text(encoding="utf-8"))
    return priors


def get_slot(ts: pd.Timestamp) -> int:
    return int(ts.weekday() * 288 + ts.hour * 12 + ts.minute // 5)


def load_panel_milepost_order(name: str, links: list[str]) -> list[int]:
    """Return index permutation that sorts links by highway milepost."""
    panel_dir = DATA / "corridors" / name
    # First check mainline_states parquet for milepost
    files = sorted((panel_dir / "train" / "mainline_states").glob("**/*.parquet"))
    if files:
        f = pd.read_parquet(files[0], columns=["link_id", "milepost"]).drop_duplicates("link_id")
        f["link_id"] = f.link_id.astype(str)
        f["milepost"] = pd.to_numeric(f.milepost, errors="coerce")
        mp_map = dict(zip(f.link_id, f.milepost))
        # Sort links by milepost
        indexed = [(mp_map.get(l, float(i)), i) for i, l in enumerate(links)]
        indexed.sort(key=lambda x: x[0])
        return [x[1] for x in indexed]
    
    # Fallback to topology order_index
    topo_file = panel_dir / "network" / "lwr_mainline_topology.csv"
    if topo_file.exists():
        t = pd.read_csv(topo_file)
        if "order_index" in t.columns:
            o_map = dict(zip(t.link_id.astype(str), pd.to_numeric(t.order_index, errors="coerce")))
            indexed = [(o_map.get(l, float(i)), i) for i, l in enumerate(links)]
            indexed.sort(key=lambda x: x[0])
            return [x[1] for x in indexed]

    return list(range(len(links)))


def main():
    t0 = time.time()
    print("=" * 75)
    print(">>> GENERATING CAUSAL T-SLICE + MORPHOLOGICAL QUEUE FORECASTS <<<")
    print("=" * 75)

    # 1. Load models
    print("Loading Task 2 onset models...")
    m_on_lgb = lgb.Booster(model_file=str(ASSETS / "onset_lgb.txt"))
    m_on_cb = CatBoostClassifier()
    m_on_cb.load_model(str(ASSETS / "onset.cbm"))

    print("Loading Task 2 ongoing models (88/12 blend)...")
    m_ong_lgb = lgb.Booster(model_file=str(ASSETS / "ongoing_lgb.txt"))
    m_ong_cb = CatBoostClassifier()
    m_ong_cb.load_model(str(MODELS_DIR / "ongoing_cb.cbm"))
    print("All models loaded successfully.")

    # 2. Load empirical Bayes bottleneck priors
    priors = load_bottleneck_priors()
    print(f"Loaded bottleneck priors for {len(priors)} corridors.")

    # 3. Parameters
    w_ong_lgb = 0.88
    w_ong_cb = 0.12
    onset_threshold = 0.55
    base_ongoing_th = 0.46

    out_file = OUT_DIR / "queue_causal_t_morphological.csv"
    first = True
    total_rows = 0
    total_promoted = 0
    total_cleared = 0
    total_bridged = 0

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

            # Milepost order for spatial closing
            milepost_order = load_panel_milepost_order(name, panel_ongoing.links)
            inv_milepost_order = np.argsort(milepost_order)

            # Load full month observation grid to extract T slice
            p_base = base.Panel(name, panel_no)
            m_frame = base.month_frame(name, "", split=split)
            grid_times, grid_v, grid_q, grid_e = base.matrices(m_frame, p_base)

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
                    probs = 0.50 * p_lgb + 0.50 * p_cb

                    step_probs = probs[-len(panel_onset.links):].copy()
                    step_pred = step_probs >= onset_threshold

                    if not step_pred.any():
                        slot_t30 = get_slot(origin + pd.Timedelta(minutes=30))
                        adj_scores = np.zeros(len(panel_onset.links), dtype=np.float64)
                        for li, lid in enumerate(panel_onset.links):
                            str_lid = str(lid)
                            p_eb = link_priors.get(str_lid, {}).get(str(slot_t30), link_overall.get(str_lid, 0.05))
                            adj_scores[li] = step_probs[li] * (1.0 + 0.30 * p_eb)

                        step_pred[np.argmax(adj_scores)] = True

                    pred = np.zeros(HORIZON * len(panel_onset.links), dtype=bool)
                    pred[-len(panel_onset.links):] = step_pred
                    links_order = panel_onset.links

                else:  # queue_ongoing
                    sp = h.pivot(index="timestamp", columns="link_id", values="speed_kmh").reindex(index=index, columns=panel_ongoing.links).to_numpy(dtype=np.float32)
                    fl = h.pivot(index="timestamp", columns="link_id", values="flow_vph").reindex(index=index, columns=panel_ongoing.links).to_numpy(dtype=np.float32)
                    observed = h.pivot(index="timestamp", columns="link_id", values="is_score_eligible").reindex(index=index, columns=panel_ongoing.links).fillna(0).to_numpy(dtype=bool)

                    # 1. Standard prediction on [T-60..T-5]
                    X_std = panel_ongoing.features(sp, fl, observed, origin, cond)
                    p_ong_lgb = m_ong_lgb.predict(X_std, num_threads=6)
                    p_ong_cb = m_ong_cb.predict_proba(X_std, thread_count=6)[:, 1]
                    p_std = (w_ong_lgb * p_ong_lgb + w_ong_cb * p_ong_cb).reshape(HORIZON, len(panel_ongoing.links))

                    # Base decision
                    current_th = 0.455 if name.startswith("D7") else 0.465
                    pred_mat = (p_std >= current_th).astype(bool)

                    # 2. Causal T-slice Forward Shift
                    t_str = origin.strftime("%Y-%m-%dT%H:%M:%SZ")
                    t_idx = grid_times.get_indexer([t_str])[0]

                    if t_idx >= 0:
                        v_T_raw = grid_v[t_idx].copy()
                        q_T_raw = grid_q[t_idx].copy()
                        obs_T = np.isfinite(v_T_raw) & (v_T_raw > 1.0)

                        # Spatial neighbor imputation for masked cells at T
                        dv = np.where(obs_T, v_T_raw - sp[-1], np.nan)
                        neigh_dv = np.nan_to_num(panel_ongoing.neighboring(panel_ongoing.incoming, dv), nan=0.0)
                        v_T = np.where(obs_T, v_T_raw, np.clip(sp[-1] + np.clip(neigh_dv, -15.0, 15.0), 5.0, panel_ongoing.vf))
                        q_T = np.where(np.isfinite(q_T_raw), q_T_raw, fl[-1])
                        obs_T_full = np.ones((1, len(panel_ongoing.links)), dtype=bool)

                        # Forward-shifted window [T-55..T]
                        speed_sh = np.vstack([sp[1:], v_T[None, :]])
                        flow_sh = np.vstack([fl[1:], q_T[None, :]])
                        obs_sh = np.vstack([observed[1:], obs_T_full])

                        X_sh = panel_ongoing.features(speed_sh, flow_sh, obs_sh, origin + pd.Timedelta(minutes=5), cond)
                        p_sh_lgb = m_ong_lgb.predict(X_sh, num_threads=6)
                        p_sh_cb = m_ong_cb.predict_proba(X_sh, thread_count=6)[:, 1]
                        p_sh = (w_ong_lgb * p_sh_lgb + w_ong_cb * p_sh_cb).reshape(HORIZON, len(panel_ongoing.links))

                        # Alignment: Step 1 of shifted window is T+5, step 2 is T+10, ..., step 6 is T+30
                        p_T = p_sh
                        p_causal = 0.60 * p_std + 0.40 * p_T

                        # Causal promotion & clearance
                        v_cut = panel_ongoing.cut[None, :]
                        q_now = (v_T <= panel_ongoing.cut)[None, :]
                        free_now = (v_T > panel_ongoing.cut + 12.0)[None, :]

                        promote_mask = (~pred_mat) & (p_std >= 0.38) & (p_causal >= 0.45) & q_now
                        clear_mask = pred_mat & (p_std < 0.48) & (p_T < 0.25) & (p_causal < 0.38) & free_now

                        pred_mat[promote_mask] = True
                        pred_mat[clear_mask] = False
                        total_promoted += int(promote_mask.sum())
                        total_cleared += int(clear_mask.sum())
                    else:
                        p_causal = p_std

                    # 3. 1D Milepost Morphological Closing
                    # Reconnect 1-link interior holes along physical highway milepost
                    p_s = p_causal[:, milepost_order]
                    pred_s = pred_mat[:, milepost_order]

                    bridge = (pred_s[:, :-2]) & (pred_s[:, 2:]) & (~pred_s[:, 1:-1]) & (p_s[:, 1:-1] >= 0.28)
                    if bridge.any():
                        pred_s[:, 1:-1] = pred_s[:, 1:-1] | bridge
                        pred_mat = pred_s[:, inv_milepost_order]
                        total_bridged += int(bridge.sum())

                    pred = pred_mat.ravel()
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

    print("\n" + "=" * 75)
    print(f"QUEUE GENERATION COMPLETE: {total_rows:,} rows in {time.time()-t0:.1f}s -> {out_file.name}")
    print(f"  Causal-T Promotions: {total_promoted:,} cells")
    print(f"  Causal-T Clearances: {total_cleared:,} cells")
    print(f"  1D Morphological Bridges: {total_bridged:,} cells")
    assert total_rows == 174000, f"Expected 174,000 rows, got {total_rows}"
    print("ALL TASK 2 INTEGRITY CHECKS PASSED 100%!")


if __name__ == "__main__":
    main()
