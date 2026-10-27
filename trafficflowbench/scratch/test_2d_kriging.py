from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import sys

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from official_toolkit.src.task1.baseline_task1_historical_mean import (
    files, slot_values, station_lanes, stable_mask, REGIMES,
    SPEED_NORMALIZER, FLOW_NORMALIZER, SPEED_WEIGHT, FLOW_WEIGHT, lane_vector
)
from official_toolkit.src.task1.baseline_task1_structural_kf import (
    smooth_deviation_rows, profile_rows
)

root = Path("data/kaggle_public")
panel = "D12_I5_N"
panel_dir = root / "corridors" / panel
lanes_by_station = station_lanes(panel_dir)

cache = np.load(PROJECT_ROOT / "outputs" / "models" / f"kf_profile_{panel}.npz", allow_pickle=True)
link_ids = [str(x) for x in cache["link_ids"]]
link_index = {lid: i for i, lid in enumerate(link_ids)}
n_links = len(link_ids)

speed_profile = {"mean": cache["speed_mean"], "fallback": cache["speed_fallback"], "link_ids": link_ids, "link_index": link_index}
flow_profile = {"mean": cache["flow_mean"], "fallback": cache["flow_fallback"], "link_ids": link_ids, "link_index": link_index}
profile_counts = {"speed_count": cache["speed_count"], "flow_count": cache["flow_count"]}

links_df = pd.read_csv(panel_dir / "network" / "links.csv")
links_df["link_id"] = links_df["link_id"].astype(str)
links_df["length_km"] = pd.to_numeric(links_df["length_km"], errors="coerce").fillna(1.0)
links_df["cum_dist"] = links_df["length_km"].cumsum()
link_dist_map = dict(zip(links_df["link_id"], links_df["cum_dist"]))
link_dist = np.array([link_dist_map.get(lid, float(i)) for i, lid in enumerate(link_ids)])

def spatial_interp_then_kalman(dev_matrix, link_dist):
    filled = dev_matrix.copy()
    for t in range(288):
        col = filled[:, t]
        finite = np.flatnonzero(np.isfinite(col))
        if len(finite) >= 2:
            x_obs = link_dist[finite]
            y_obs = col[finite]
            f = interp1d(x_obs, y_obs, kind="linear", bounds_error=False, fill_value=(y_obs[0], y_obs[-1]))
            filled[:, t] = f(link_dist)
        elif len(finite) == 1:
            filled[:, t] = col[finite[0]]
            
    return smooth_deviation_rows(filled)

val_files = files(panel_dir, "validation")
print(f"Evaluating 1D Kalman vs 2D Spatiotemporal on ALL {len(val_files)} validation days...")

sums_1d = {r: {"s_sq": 0.0, "q_sq": 0.0, "n": 0} for r in REGIMES}
sums_2d = {r: {"s_sq": 0.0, "q_sq": 0.0, "n": 0} for r in REGIMES}

for path in val_files:
    frame = pd.read_parquet(
        path,
        columns=["date", "timestamp", "station_id", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"],
    )
    frame["link_id"] = frame.link_id.astype(str)
    li = frame.link_id.map(link_index).fillna(-1).to_numpy(dtype=np.int64)
    known = li >= 0
    safe_li = np.where(known, li, 0)
    weekday, tod = slot_values(frame)
    slot = weekday * 288 + tod
    true_speed = pd.to_numeric(frame.speed_kmh, errors="coerce").to_numpy(dtype=float)
    true_flow = pd.to_numeric(frame.flow_vph, errors="coerce").to_numpy(dtype=float)
    eligible = frame.is_score_eligible.astype(bool).to_numpy().copy()
    prof_speed = profile_rows(speed_profile, profile_counts["speed_count"], safe_li, slot)
    prof_flow = profile_rows(flow_profile, profile_counts["flow_count"], safe_li, slot)
    
    for regime in REGIMES:
        target = eligible & stable_mask(panel, regime, frame.date, frame.timestamp, frame.link_id)
        if not target.any():
            continue
            
        dev_speed = np.full((n_links, 288), np.nan)
        dev_flow = np.full((n_links, 288), np.nan)
        dev_speed_n = np.zeros((n_links, 288), dtype=np.int32)
        dev_flow_n = np.zeros((n_links, 288), dtype=np.int32)
        dev_speed_sum = np.zeros((n_links, 288), dtype=float)
        dev_flow_sum = np.zeros((n_links, 288), dtype=float)
        
        obs_speed = eligible & ~target & known & np.isfinite(true_speed) & np.isfinite(prof_speed)
        obs_flow = eligible & ~target & known & np.isfinite(true_flow) & np.isfinite(prof_flow)
        np.add.at(dev_speed_sum, (li[obs_speed], tod[obs_speed]), (true_speed - prof_speed)[obs_speed])
        np.add.at(dev_flow_sum, (li[obs_flow], tod[obs_flow]), (true_flow - prof_flow)[obs_flow])
        np.add.at(dev_speed_n, (li[obs_speed], tod[obs_speed]), 1)
        np.add.at(dev_flow_n, (li[obs_flow], tod[obs_flow]), 1)
        np.divide(dev_speed_sum, dev_speed_n, out=dev_speed, where=dev_speed_n > 0)
        np.divide(dev_flow_sum, dev_flow_n, out=dev_flow, where=dev_flow_n > 0)
        
        # 1D time-only Kalman
        smooth_s_1d = smooth_deviation_rows(dev_speed)
        smooth_q_1d = smooth_deviation_rows(dev_flow)
        pred_s_1d = np.clip(prof_speed + smooth_s_1d[safe_li, tod], 0.0, None)
        pred_q_1d = np.clip(prof_flow + smooth_q_1d[safe_li, tod], 0.0, None)
        
        # 2D spatiotemporal
        smooth_s_2d = spatial_interp_then_kalman(dev_speed, link_dist)
        smooth_q_2d = spatial_interp_then_kalman(dev_flow, link_dist)
        pred_s_2d = np.clip(prof_speed + smooth_s_2d[safe_li, tod], 0.0, None)
        pred_q_2d = np.clip(prof_flow + smooth_q_2d[safe_li, tod], 0.0, None)
        
        mask = target & np.isfinite(true_speed) & np.isfinite(true_flow)
        lanes_vec = lane_vector(frame.station_id, frame.link_id, lanes_by_station)[mask]
        
        ds_1d = pred_s_1d[mask] - true_speed[mask]
        dq_1d = (pred_q_1d[mask] - true_flow[mask]) / lanes_vec
        sums_1d[regime]["s_sq"] += float(np.sum(ds_1d * ds_1d))
        sums_1d[regime]["q_sq"] += float(np.sum(dq_1d * dq_1d))
        sums_1d[regime]["n"] += int(mask.sum())
        
        ds_2d = pred_s_2d[mask] - true_speed[mask]
        dq_2d = (pred_q_2d[mask] - true_flow[mask]) / lanes_vec
        sums_2d[regime]["s_sq"] += float(np.sum(ds_2d * ds_2d))
        sums_2d[regime]["q_sq"] += float(np.sum(dq_2d * dq_2d))
        sums_2d[regime]["n"] += int(mask.sum())

print("\n--- Final Results on Validation Split ---")
scores_1d = []
scores_2d = []
for r in REGIMES:
    n = max(sums_1d[r]["n"], 1)
    rmse_s_1d = np.sqrt(sums_1d[r]["s_sq"] / n)
    rmse_q_1d = np.sqrt(sums_1d[r]["q_sq"] / n)
    s_state_1d = SPEED_WEIGHT * max(0, 1 - rmse_s_1d / SPEED_NORMALIZER) + FLOW_WEIGHT * max(0, 1 - rmse_q_1d / FLOW_NORMALIZER)
    scores_1d.append(s_state_1d)
    
    rmse_s_2d = np.sqrt(sums_2d[r]["s_sq"] / n)
    rmse_q_2d = np.sqrt(sums_2d[r]["q_sq"] / n)
    s_state_2d = SPEED_WEIGHT * max(0, 1 - rmse_s_2d / SPEED_NORMALIZER) + FLOW_WEIGHT * max(0, 1 - rmse_q_2d / FLOW_NORMALIZER)
    scores_2d.append(s_state_2d)
    
    print(f"[{r}] 1D Kalman S_state: {s_state_1d:.4f} (RMSE: s={rmse_s_1d:.2f}, q={rmse_q_1d:.2f})")
    print(f"[{r}] 2D Spatio  S_state: {s_state_2d:.4f} (RMSE: s={rmse_s_2d:.2f}, q={rmse_q_2d:.2f})")

print(f"\nMacro-Mean 1D Kalman S_state: {np.mean(scores_1d):.4f}")
print(f"Macro-Mean 2D Spatio  S_state: {np.mean(scores_2d):.4f}")
