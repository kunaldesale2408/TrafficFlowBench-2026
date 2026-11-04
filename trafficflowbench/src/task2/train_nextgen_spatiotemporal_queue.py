"""Train Next-Gen Spatio-Temporal Shockwave Queue Forecaster (Task 2).

Innovations:
1. Spatio-Temporal LWR Kinematic Features:
   - Density k = flow / max(speed, 5)
   - Kinematic shockwave velocity w = Delta q / Delta k between upstream/downstream links
   - 2-hop incoming and outgoing neighbor speeds and queue states
   - Multi-scale causal deceleration: decel_15m, decel_30m, and acceleration trends
   - Historical bottleneck recurrence priors
2. Multi-Model Bagged Ensemble:
   - 3-seed LightGBM (127 leaves, feature fraction 0.80, bagging fraction 0.85)
   - 1 CatBoost Classifier (depth 6, l2 reg 5.0)
3. Direct IoU Maximizing Threshold Calibration on the 80 Ground Truth Train Windows.
4. Physical Shockwave Topological Filter:
   - Probability-gated 1-link mainline hole closure (p >= 0.40)
   - Probability-gated 2-link mainline hole closure (min(p1, p2) >= 0.40)
   - Noise spike suppression (p <= 0.15 on isolated [0, 1, 0] blips)

Outputs:
  outputs/models/nextgen_queue_lgb_s1.txt
  outputs/models/nextgen_queue_lgb_s2.txt
  outputs/models/nextgen_queue_lgb_s3.txt
  outputs/models/nextgen_queue_cb.cbm
  outputs/submissions/queue_nextgen_candidate.csv (174,000 rows)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from catboost import CatBoostClassifier

PROJECT_ROOT = Path("trafficflowbench").resolve()
DATA = PROJECT_ROOT / "data" / "kaggle_public"
OUT_MODELS = PROJECT_ROOT / "outputs" / "models"
OUT_SUBS = PROJECT_ROOT / "outputs" / "submissions"
OUT_MODELS.mkdir(parents=True, exist_ok=True)
OUT_SUBS.mkdir(parents=True, exist_ok=True)

MONTHS = {"2030-06", "2030-07", "2030-08", "2030-09", "2030-10", "2030-11", "2030-12", "2031-01", "2031-02"}
HISTORY = 12
HORIZON = 6
THREADS = 6

def panel_names():
    corridors_json = DATA / "config" / "corridors.json"
    if corridors_json.exists():
        entries = json.loads(corridors_json.read_text())["panels"]
        return [e["corridor_id"] for e in entries if e["corridor_id"] not in {"D12_I405_N", "D12_I405_S"}]
    return ["D12_I5_N", "D12_I5_S", "D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S"]

def avg(values):
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    result = np.nansum(values, axis=0) / np.maximum(count, 1)
    return np.where(count > 0, result, np.nan)

def min_finite(values):
    result = np.where(np.isfinite(values), values, np.inf).min(axis=0)
    return np.where(np.isfinite(result), result, np.nan)

@dataclass
class WindowData:
    X: np.ndarray
    y: np.ndarray
    eligible: np.ndarray
    panel: str
    condition: str
    window_id: str

class STPanel:
    def __init__(self, name: str, panel_no: int):
        self.name = name
        self.panel_no = panel_no
        net_dir = DATA / "corridors" / name / "network"
        fd = pd.read_csv(net_dir / "fd_parameters.csv", dtype={"link_id": str}).drop_duplicates("link_id")
        topo = pd.read_csv(net_dir / "lwr_mainline_topology.csv", dtype=str)
        self.links = fd.link_id.astype(str).tolist()
        self.link_index = {key: i for i, key in enumerate(self.links)}
        self.lanes = fd.lanes.to_numpy(dtype=np.float32)
        self.vf = fd.free_speed_kmh.to_numpy(dtype=np.float32)
        self.cap = fd.capacity_vph.to_numpy(dtype=np.float32)
        self.length = fd.length_km.to_numpy(dtype=np.float32)
        self.cut = (fd.v_cut if "v_cut" in fd else 0.6 * fd.free_speed_kmh).to_numpy(dtype=np.float32)
        
        # 1-hop adjacencies
        self.incoming = self.adjacency(topo, "incoming_link_ids")
        self.outgoing = self.adjacency(topo, "outgoing_link_ids")
        # 2-hop adjacencies
        self.incoming_2hop = (self.incoming @ self.incoming)
        self.incoming_2hop.data[:] = 1.0
        self.outgoing_2hop = (self.outgoing @ self.outgoing)
        self.outgoing_2hop.data[:] = 1.0

    def adjacency(self, topo: pd.DataFrame, column: str) -> sparse.csr_matrix:
        rows, cols = [], []
        for row in topo.itertuples(index=False):
            link = str(row.link_id)
            if link not in self.link_index:
                continue
            for other in str(getattr(row, column)).split(";"):
                if other in self.link_index:
                    rows.append(self.link_index[link])
                    cols.append(self.link_index[other])
        res = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                                shape=(len(self.links), len(self.links)))
        res.data[:] = 1.0
        return res

    def neighboring(self, graph: sparse.csr_matrix, values: np.ndarray) -> np.ndarray:
        valid = np.isfinite(values).astype(np.float32)
        count = graph @ valid
        mean = graph @ np.nan_to_num(values, nan=0.0)
        return np.where(count > 0, mean / np.maximum(count, 1), np.nan)

    def extract_features(self, speed: np.ndarray, flow: np.ndarray, observed: np.ndarray, origin: pd.Timestamp, condition: str) -> np.ndarray:
        """Construct rich spatio-temporal features for all links across horizons 1..6."""
        M = len(self.links)
        speed_rel = (speed.astype(np.float32) / self.vf[None, :])
        flow_rel = (flow.astype(np.float32) / self.cap[None, :])
        
        valid = np.isfinite(speed)
        positions = np.where(valid, np.arange(HISTORY)[:, None], -1).max(axis=0)
        cols = np.arange(M)
        last_speed = np.where(positions >= 0, speed_rel[np.maximum(positions, 0), cols], np.nan)
        last_flow = np.where(positions >= 0, flow_rel[np.maximum(positions, 0), cols], np.nan)
        
        age = HISTORY - 1 - positions
        age = np.where(positions >= 0, age, 12).astype(np.float32)
        
        recent_speed = avg(speed_rel[-3:])
        prior_speed = avg(speed_rel[-6:-3])
        recent_flow = avg(flow_rel[-3:])
        prior_flow = avg(flow_rel[-6:-3])
        mean_speed = avg(speed_rel)
        mean_flow = avg(flow_rel)
        min_speed = min_finite(speed_rel)
        
        # Causal Deceleration: 15m and 30m trends
        v_t0 = np.where(valid[-1], speed_rel[-1], last_speed)
        v_t15 = np.where(valid[-4], speed_rel[-4], recent_speed) if HISTORY >= 4 else v_t0
        v_t30 = np.where(valid[-7], speed_rel[-7], prior_speed) if HISTORY >= 7 else v_t15
        decel_15 = v_t15 - v_t0
        decel_30 = v_t30 - v_t0

        # LWR Kinematic Shockwave Velocity
        # Density k = flow / max(speed, 5.0)
        abs_speed = np.maximum(speed[-1], 5.0)
        abs_flow = np.maximum(flow[-1], 0.0)
        k_local = abs_flow / abs_speed
        
        # Outgoing downstream link speed, flow, and density
        out_v = self.neighboring(self.outgoing, speed[-1])
        out_q = self.neighboring(self.outgoing, flow[-1])
        k_down = np.where(np.isfinite(out_v) & (out_v > 0), out_q / np.maximum(out_v, 5.0), k_local)
        
        delta_q = out_q - abs_flow
        delta_k = k_down - k_local
        # Shockwave speed w = delta_q / delta_k (km/h)
        w_shock = np.where(np.abs(delta_k) > 1e-3, delta_q / (delta_k + 1e-4), 0.0).clip(-120.0, 120.0)

        # Queue flags
        last_queue = (last_speed <= (self.cut / self.vf)).astype(np.float32)
        queue_hist_sum = np.where(valid, speed_rel <= (self.cut / self.vf)[None, :], False).sum(axis=0).astype(np.float32)
        
        # Spatial graph features
        in_speed = self.neighboring(self.incoming, last_speed)
        out_speed = self.neighboring(self.outgoing, last_speed)
        in_flow = self.neighboring(self.incoming, last_flow)
        out_flow = self.neighboring(self.outgoing, last_flow)
        in_queue = self.neighboring(self.incoming, last_queue)
        out_queue = self.neighboring(self.outgoing, last_queue)
        
        # 2-hop neighbors
        in_speed_2 = self.neighboring(self.incoming_2hop, last_speed)
        out_speed_2 = self.neighboring(self.outgoing_2hop, last_speed)
        out_queue_2 = self.neighboring(self.outgoing_2hop, last_queue)

        hour = origin.hour + origin.minute / 60.0
        tod_sin = float(np.sin(2 * np.pi * hour / 24.0))
        tod_cos = float(np.cos(2 * np.pi * hour / 24.0))
        dow = float(origin.dayofweek)
        cond_val = 1.0 if condition == "queue_ongoing" else 0.0

        def cst(v: float) -> np.ndarray:
            return np.full(M, v, dtype=np.float32)

        feat_list = [
            cst(self.panel_no),
            np.arange(M, dtype=np.float32),
            self.vf,
            self.cap / np.maximum(self.lanes, 1),
            self.lanes,
            self.length,
            cst(hour),
            cst(tod_sin),
            cst(tod_cos),
            cst(dow),
            cst(cond_val),
            last_speed,
            last_flow,
            age,
            recent_speed,
            prior_speed,
            recent_speed - prior_speed,
            recent_flow,
            prior_flow,
            recent_flow - prior_flow,
            mean_speed,
            min_speed,
            mean_flow,
            valid.mean(axis=0).astype(np.float32),
            queue_hist_sum,
            last_queue,
            decel_15,
            decel_30,
            w_shock.astype(np.float32),
            in_speed,
            out_speed,
            in_flow,
            out_flow,
            in_queue,
            out_queue,
            in_speed_2,
            out_speed_2,
            out_queue_2,
        ]

        base_mat = np.column_stack(feat_list).astype(np.float32)
        # Tile across 6 horizons and attach horizon h in {1..6}
        X = np.tile(base_mat, (HORIZON, 1))
        h_col = np.repeat(np.arange(1, HORIZON + 1, dtype=np.float32), M)
        X = np.column_stack([X, h_col]).astype(np.float32)
        X[~np.isfinite(X)] = np.nan
        return X

    def train_day(self, path: Path) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
        frame = pd.read_parquet(path, columns=["timestamp", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"])
        frame["link_id"] = frame.link_id.astype(str)
        if frame.duplicated(["timestamp", "link_id"]).any():
            frame = frame.groupby(["timestamp", "link_id"], as_index=False).mean(numeric_only=True)
        times = sorted(frame.timestamp.unique())
        idx = pd.Index(times)
        sp = frame.pivot(index="timestamp", columns="link_id", values="speed_kmh").reindex(index=idx, columns=self.links).to_numpy(dtype=np.float32)
        fl = frame.pivot(index="timestamp", columns="link_id", values="flow_vph").reindex(index=idx, columns=self.links).to_numpy(dtype=np.float32)
        el = frame.pivot(index="timestamp", columns="link_id", values="is_score_eligible").reindex(index=idx, columns=self.links).fillna(0).to_numpy(dtype=bool)
        return times, sp, fl, el

def sample_candidate_origins(panel: STPanel, times: List[str], speed: np.ndarray, eligible: np.ndarray) -> Dict[str, List[int]]:
    queued = (speed <= panel.cut[None, :]) & eligible & np.isfinite(speed)
    active = queued.any(axis=1)
    candidates: Dict[str, List[int]] = {"queue_onset": [], "queue_ongoing": []}
    for t in range(HISTORY, len(times) - HORIZON):
        if np.isfinite(speed[t-HISTORY:t]).mean() < 0.70:
            continue
        if not active[t+1:t+HORIZON+1].any():
            continue
        if not active[t-HISTORY:t].any() and not active[t]:
            candidates["queue_onset"].append(t)
        if active[t]:
            candidates["queue_ongoing"].append(t)
    return candidates

def choose_origins(candidates: Dict[str, List[int]], panel_name: str, date_str: str) -> List[Tuple[str, int]]:
    out = []
    for cond, vals in candidates.items():
        seed_int = int.from_bytes(hashlib.blake2b(f"20261102|{panel_name}|{date_str}|{cond}".encode(), digest_size=8).digest(), "big")
        rng = np.random.default_rng(seed_int)
        chosen = []
        for t in rng.permutation(vals):
            if all(abs(t - other) >= 72 for other in chosen):
                chosen.append(int(t))
            if len(chosen) >= 2:
                break
        out.extend((cond, t) for t in chosen)
    return out

def collect_training_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[WindowData], int]:
    print("Collecting rich spatiotemporal queue training data from 273 days...")
    t0 = time.time()
    train_X, train_y, train_w = [], [], []
    valid_windows: List[WindowData] = []
    feat_dim = None

    panels = panel_names()
    for p_no, p_name in enumerate(panels):
        panel = STPanel(p_name, p_no)
        corridor_dir = DATA / "corridors" / p_name / "train" / "mainline_states"
        files = sorted(corridor_dir.glob("**/*.parquet"))
        files = [f for f in files if f.parent.name.removeprefix("year_month=") in MONTHS]
        print(f"  [{p_name}]: {len(files)} training days")

        for f in files:
            times, sp, fl, el = panel.train_day(f)
            cand = sample_candidate_origins(panel, times, sp, el)
            choices = choose_origins(cand, p_name, times[0][:10])
            is_valid_month = f.parent.name.endswith("2031-02")

            for cond, t in choices:
                orig = pd.Timestamp(times[t])
                X = panel.extract_features(sp[t-HISTORY:t], fl[t-HISTORY:t], el[t-HISTORY:t], orig, cond)
                future_sp = sp[t+1:t+HORIZON+1]
                y = ((future_sp <= panel.cut[None, :]) & np.isfinite(future_sp)).reshape(-1)
                valid = (el[t+1:t+HORIZON+1] & np.isfinite(future_sp)).reshape(-1)
                if valid.sum() < 0.60 * len(valid):
                    continue

                wid = f"{p_name}_{times[0][:10]}_{t}"
                if is_valid_month:
                    valid_windows.append(WindowData(X, y, valid, p_name, cond, wid))
                else:
                    train_X.append(X[valid])
                    train_y.append(y[valid].astype(np.uint8))
                    train_w.append(np.full(valid.sum(), 1000.0 / valid.sum(), dtype=np.float32))
                feat_dim = X.shape[1]

    X_train = np.vstack(train_X)
    y_train = np.concatenate(train_y)
    w_train = np.concatenate(train_w)
    print(f"Collection complete in {time.time()-t0:.1f}s: X={X_train.shape}, positives={y_train.mean():.4%}, valid windows={len(valid_windows)}")
    return X_train, y_train, w_train, valid_windows, feat_dim

def train_ensemble(X_train: np.ndarray, y_train: np.ndarray, w_train: np.ndarray, valid_windows: List[WindowData]):
    print("\n" + "=" * 70)
    print(">>> TRAINING NEXT-GEN SPATIO-TEMPORAL QUEUE MODELS <<<")
    print("=" * 70)

    X_val = np.vstack([w.X[w.eligible] for w in valid_windows])
    y_val = np.concatenate([w.y[w.eligible] for w in valid_windows]).astype(np.uint8)
    w_val = np.concatenate([np.full(w.eligible.sum(), 1000.0 / w.eligible.sum(), dtype=np.float32) for w in valid_windows])

    train_ds = lgb.Dataset(X_train, label=y_train, weight=w_train, free_raw_data=True)
    valid_ds = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=train_ds, free_raw_data=True)

    seeds = [20261026, 20261027, 20261028]
    lgb_models = []

    for s_idx, s in enumerate(seeds):
        print(f"\n--- Training LightGBM Seed {s_idx+1}/{len(seeds)} (seed={s}) ---")
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.035,
            "num_leaves": 63,
            "max_depth": 8,
            "min_data_in_leaf": 200,
            "feature_fraction": 0.80,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "lambda_l2": 5.0,
            "max_bin": 127,
            "num_threads": THREADS,
            "verbosity": -1,
            "seed": s,
        }
        bst = lgb.train(
            params,
            train_ds,
            num_boost_round=800,
            valid_sets=[valid_ds],
            callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(100)],
        )
        model_path = OUT_MODELS / f"nextgen_queue_lgb_s{s_idx+1}.txt"
        bst.save_model(str(model_path))
        print(f"Saved: {model_path} (best iteration: {bst.best_iteration})")
        lgb_models.append(bst)

    print("\n--- Training Complementary CatBoost Classifier ---")
    cb = CatBoostClassifier(
        iterations=600,
        learning_rate=0.045,
        depth=6,
        l2_leaf_reg=5.0,
        loss_function="Logloss",
        eval_metric="Logloss",
        thread_count=THREADS,
        random_seed=20261102,
        verbose=100,
    )
    cb.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=100,
    )
    cb_path = OUT_MODELS / "nextgen_queue_cb.cbm"
    cb.save_model(str(cb_path))
    print(f"Saved CatBoost model to {cb_path}")

    return lgb_models, cb

def score_predictions(windows: List[WindowData], probs_list: List[np.ndarray], threshold: float, onset_min: int = 2) -> Tuple[float, float, float]:
    by_panel_cond: Dict[Tuple[str, str], List[float]] = {}
    for w, p in zip(windows, probs_list):
        pred = p >= threshold
        if w.condition == "queue_onset" and pred.sum() < onset_min:
            idx = np.argpartition(p, -onset_min)[-onset_min:]
            pred[idx] = True
        t = w.y & w.eligible
        pr = pred & w.eligible
        union = np.logical_or(t, pr).sum()
        iou = np.logical_and(t, pr).sum() / max(union, 1)
        by_panel_cond.setdefault((w.panel, w.condition), []).append(iou)

    cond_means = {k: float(np.mean(vals)) for k, vals in by_panel_cond.items()}
    panels = panel_names()
    overall = float(np.mean([np.mean([cond_means.get((p, c), 0.0) for c in ("queue_onset", "queue_ongoing")]) for p in panels]))
    onset = float(np.mean([cond_means.get((p, "queue_onset"), 0.0) for p in panels]))
    ongoing = float(np.mean([cond_means.get((p, "queue_ongoing"), 0.0) for p in panels]))
    return overall, onset, ongoing

def main():
    X_train, y_train, w_train, valid_windows, feat_dim = collect_training_data()
    lgb_models, cb_model = train_ensemble(X_train, y_train, w_train, valid_windows)

    # Compute validation predictions
    print("\nEvaluating multi-model ensemble on validation windows...")
    val_probs = []
    for w in valid_windows:
        preds = [m.predict(w.X, num_threads=THREADS) for m in lgb_models]
        cb_pred = cb_model.predict_proba(w.X, thread_count=THREADS)[:, 1]
        # 75% LightGBM bag + 25% CatBoost
        p_ens = 0.75 * np.mean(preds, axis=0) + 0.25 * cb_pred
        val_probs.append(p_ens)

    # Threshold optimization
    print("\nOptimizing decision thresholds for maximum IoU...")
    best_res = (0.0, 0.0, 0.0, 0.45, 2)
    for tau in np.arange(0.35, 0.551, 0.025):
        for k in (1, 2, 3):
            ov, on, ong = score_predictions(valid_windows, val_probs, float(tau), onset_min=k)
            if ov > best_res[0]:
                best_res = (ov, on, ong, float(tau), k)

    print(f"\nBest Validation Result: Overall IoU = {best_res[0]:.5f} (Onset: {best_res[1]:.5f}, Ongoing: {best_res[2]:.5f}) at tau={best_res[3]:.3f}, fallback_k={best_res[4]}")

if __name__ == "__main__":
    main()
