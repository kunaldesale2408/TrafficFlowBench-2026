"""Task 1 — Structural State-Space Kalman Reconstruction with FD Bounds.

Combines:
1. Long-term regular pattern: 273-day historical profile (weekday x 5-minute time-of-day slot).
2. Intra-day dynamic pattern: Local-level Kalman filter with Rauch-Tung-Striebel (RTS)
   backward smoothing on unmasked same-day deviations (Zhou & Mahmassani, 2007).
3. Physical Fundamental Diagram compliance:
   - Speed clipped to [5.0, free_speed_kmh]
   - Flow clipped to [50.0, capacity_vph] (satisfies non-empty road threshold)
   - Density bounded by jam_density: flow <= jam_density * speed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from official_toolkit.src.task1.baseline_task1_historical_mean import (
    DEFAULT_RELEASE,
    build_profile,
    check_release_root,
    masked_files,
    slot_values,
)
from official_toolkit.src.task1.baseline_task1_structural_kf import (
    smooth_deviation_rows,
    profile_rows,
)


OUTPUT_COLUMNS = ["panel", "timestamp", "station_id", "link_id", "mask_regime", "speed_kmh", "flow_vph"]


def load_link_fd_bounds(panel_dir: Path) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Load free speed, capacity, and jam density bounds per link."""
    links_path = panel_dir / "network" / "links.csv"
    fd_path = panel_dir / "network" / "fd_parameters.csv"

    ffs_map = {}
    cap_map = {}
    jam_map = {}

    if links_path.exists():
        links = pd.read_csv(links_path)
        for _, row in links.iterrows():
            lid = str(row["link_id"])
            if "free_speed_kmh" in row and pd.notna(row["free_speed_kmh"]):
                ffs_map[lid] = float(row["free_speed_kmh"])
            lanes = float(row.get("lanes", 3))
            cap = float(row.get("capacity_vph", lanes * 2000.0))
            cap_map[lid] = cap

    if fd_path.exists():
        fd = pd.read_csv(fd_path)
        for _, row in fd.iterrows():
            lid = str(row["link_id"])
            if "v_free" in row and pd.notna(row["v_free"]):
                ffs_map[lid] = float(row["v_free"])
            if "q_cap" in row and pd.notna(row["q_cap"]):
                cap_map[lid] = float(row["q_cap"])
            if "k_jam" in row and pd.notna(row["k_jam"]):
                jam_map[lid] = float(row["k_jam"])

    return ffs_map, cap_map, jam_map


def spatial_interp_then_kalman(dev_matrix: np.ndarray, link_dist: np.ndarray) -> np.ndarray:
    """Interpolate spatial gaps across freeway coordinates, then smooth in time."""
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
        else:
            filled[:, t] = 0.0

    return smooth_deviation_rows(filled)


def build_panel_kf_submission(
    panel: str,
    release: Path,
    splits: List[str],
    output: Path,
    write_header: bool,
) -> int:
    """Build Kalman-smoothed predictions for one corridor across target splits."""
    check_release_root(release, panel)
    panel_dir = release / "corridors" / panel

    print(f"\n[Task 1 KF] Corridor: {panel}")
    # 1. Compute or load 273-day historical profile
    profile_cache = PROJECT_ROOT / "outputs" / "models" / f"kf_profile_{panel}.npz"
    if profile_cache.exists():
        print(f"  Loading cached profile from {profile_cache.name}...")
        data = np.load(profile_cache, allow_pickle=True)
        link_ids = [str(x) for x in data["link_ids"]]
        link_index = {lid: i for i, lid in enumerate(link_ids)}
        speed_profile = {
            "mean": data["speed_mean"],
            "fallback": data["speed_fallback"],
            "link_ids": link_ids,
            "link_index": link_index,
        }
        flow_profile = {
            "mean": data["flow_mean"],
            "fallback": data["flow_fallback"],
            "link_ids": link_ids,
            "link_index": link_index,
        }
        profile_counts = {
            "speed_count": data["speed_count"],
            "flow_count": data["flow_count"],
        }
    else:
        print(f"  Computing historical 273-day profile from train split...")
        speed_profile, flow_profile, profile_counts = build_profile(panel, panel_dir)
        profile_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            profile_cache,
            link_ids=speed_profile["link_ids"],
            speed_mean=speed_profile["mean"],
            speed_fallback=speed_profile["fallback"],
            flow_mean=flow_profile["mean"],
            flow_fallback=flow_profile["fallback"],
            speed_count=profile_counts["speed_count"],
            flow_count=profile_counts["flow_count"],
        )
        link_ids = speed_profile["link_ids"]
        link_index = speed_profile["link_index"]

    n_links = len(link_ids)
    ffs_map, cap_map, jam_map = load_link_fd_bounds(panel_dir)

    links_path = panel_dir / "network" / "links.csv"
    link_dist_map = {}
    if links_path.exists():
        links_df = pd.read_csv(links_path)
        links_df["link_id"] = links_df["link_id"].astype(str)
        l_col = "length_km" if "length_km" in links_df.columns else ("length_m" if "length_m" in links_df.columns else None)
        if l_col:
            lengths = pd.to_numeric(links_df[l_col], errors="coerce").fillna(1.0)
            if l_col == "length_m":
                lengths = lengths / 1000.0
            links_df["cum_dist"] = lengths.cumsum()
            link_dist_map = dict(zip(links_df["link_id"], links_df["cum_dist"]))
    link_dist = np.array([link_dist_map.get(lid, float(i)) for i, lid in enumerate(link_ids)])

    rows_written = 0
    header_needed = write_header

    for split in splits:
        # Load official submission template keys for this panel & split
        tmpl_file = release / "task1" / panel / split / "sample_submission_state.csv"
        if not tmpl_file.exists():
            tmpl_file = release / "submission_templates_per_task" / f"sample_submission_state_{panel}_{split}.csv"
        if not tmpl_file.exists():
            continue

        template = pd.read_csv(
            tmpl_file,
            usecols=["timestamp", "station_id", "link_id"],
            dtype=str,
        )
        template_keys = set(zip(template.timestamp, template.station_id, template.link_id))
        print(f"  Split '{split}': processing {len(template_keys):,} target cells...")

        for path in masked_files(panel_dir, split):
            frame = pd.read_parquet(
                path,
                columns=[
                    "date", "timestamp", "station_id", "link_id", "speed_kmh",
                    "flow_vph", "is_score_eligible", "mask_regime",
                ],
            )
            frame["station_id"] = frame.station_id.astype(str)
            frame["link_id"] = frame.link_id.astype(str)
            weekday, tod = slot_values(frame)
            slot = weekday * 288 + tod

            li = frame.link_id.map(link_index).fillna(-1).to_numpy(dtype=np.int64)
            known = li >= 0
            safe_li = np.where(known, li, 0)

            prof_speed = profile_rows(speed_profile, profile_counts["speed_count"], safe_li, slot)
            prof_flow = profile_rows(flow_profile, profile_counts["flow_count"], safe_li, slot)

            true_speed = pd.to_numeric(frame.speed_kmh, errors="coerce").to_numpy(dtype=float)
            true_flow = pd.to_numeric(frame.flow_vph, errors="coerce").to_numpy(dtype=float)
            eligible = frame.is_score_eligible.astype(bool).to_numpy()

            # Target cells are those blanked in the template
            blanked = np.fromiter(
                (k in template_keys for k in zip(
                    frame.timestamp.astype(str),
                    frame.station_id.astype(str),
                    frame.link_id.astype(str),
                )),
                dtype=bool,
                count=len(frame),
            )

            for regime in pd.unique(frame.mask_regime.astype(str)):
                regime_mask = (frame.mask_regime.astype(str) == regime).to_numpy()
                target = blanked & regime_mask

                dev_speed = np.full((n_links, 288), np.nan)
                dev_flow = np.full((n_links, 288), np.nan)
                dev_speed_n = np.zeros((n_links, 288), dtype=np.int32)
                dev_flow_n = np.zeros((n_links, 288), dtype=np.int32)
                dev_speed_sum = np.zeros((n_links, 288), dtype=float)
                dev_flow_sum = np.zeros((n_links, 288), dtype=float)

                # Unmasked cells for same-day deviation estimation
                obs_speed = eligible & regime_mask & ~blanked & known & np.isfinite(true_speed) & np.isfinite(prof_speed)
                obs_flow = eligible & regime_mask & ~blanked & known & np.isfinite(true_flow) & np.isfinite(prof_flow)

                np.add.at(dev_speed_sum, (li[obs_speed], tod[obs_speed]), (true_speed - prof_speed)[obs_speed])
                np.add.at(dev_flow_sum, (li[obs_flow], tod[obs_flow]), (true_flow - prof_flow)[obs_flow])
                np.add.at(dev_speed_n, (li[obs_speed], tod[obs_speed]), 1)
                np.add.at(dev_flow_n, (li[obs_flow], tod[obs_flow]), 1)

                np.divide(dev_speed_sum, dev_speed_n, out=dev_speed, where=dev_speed_n > 0)
                np.divide(dev_flow_sum, dev_flow_n, out=dev_flow, where=dev_flow_n > 0)

                # 1D Temporal Kalman filter + RTS smoother
                smooth_speed_1d = smooth_deviation_rows(dev_speed)
                smooth_flow_1d = smooth_deviation_rows(dev_flow)

                # 2D Spatiotemporal deviation smoothing across freeway coordinate
                smooth_speed_2d = spatial_interp_then_kalman(dev_speed, link_dist)
                smooth_flow_2d = spatial_interp_then_kalman(dev_flow, link_dist)

                # Optimal blend (60% temporal + 40% spatial)
                smooth_speed = 0.60 * smooth_speed_1d + 0.40 * smooth_speed_2d
                smooth_flow = 0.60 * smooth_flow_1d + 0.40 * smooth_flow_2d

                # Combine profile + smoothed intra-day deviation
                pred_speed = prof_speed + smooth_speed[safe_li, tod]
                pred_flow = prof_flow + smooth_flow[safe_li, tod]
                pred_speed = np.where(known, pred_speed, prof_speed)
                pred_flow = np.where(known, pred_flow, prof_flow)

                # Target mask for output
                mask = target & np.isfinite(pred_speed) & np.isfinite(pred_flow)
                if not mask.any():
                    continue

                sub_links = frame.loc[mask, "link_id"].to_numpy()
                sub_speeds = pred_speed[mask]
                sub_flows = pred_flow[mask]

                # Physical FD bounds enforcement
                ffs_vals = np.array([ffs_map.get(l, 115.0) for l in sub_links])
                cap_vals = np.array([cap_map.get(l, 10000.0) for l in sub_links])
                jam_vals = np.array([jam_map.get(l, 450.0) for l in sub_links])

                sub_speeds = np.clip(sub_speeds, 5.0, ffs_vals)
                sub_flows = np.clip(sub_flows, 100.0, cap_vals)

                # Jam density constraint
                implied_k = sub_flows / np.maximum(sub_speeds, 1.0)
                excess = implied_k > jam_vals
                if np.any(excess):
                    sub_flows[excess] = jam_vals[excess] * sub_speeds[excess]

                out = pd.DataFrame(
                    {
                        "panel": panel,
                        "timestamp": frame.loc[mask, "timestamp"].astype(str).to_numpy(),
                        "station_id": frame.loc[mask, "station_id"].to_numpy(),
                        "link_id": sub_links,
                        "mask_regime": regime,
                        "speed_kmh": sub_speeds,
                        "flow_vph": sub_flows,
                    },
                    columns=OUTPUT_COLUMNS,
                )

                out.to_csv(output, mode="w" if header_needed else "a", header=header_needed, index=False)
                header_needed = False
                rows_written += len(out)

    return rows_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 1 Structural State-Space Kalman Submission Pipeline")
    parser.add_argument("--release-root", type=Path, default=Path("./data/kaggle_public"))
    parser.add_argument("--split", choices=["train", "validation", "private", "all"], default="all")
    parser.add_argument("--output", type=Path, default=Path("outputs/submissions/state_submission.csv"))
    parser.add_argument("--panel", action="append", help="optional panel filter")
    args = parser.parse_args()

    root = args.release_root.resolve()
    splits = ["validation", "private"] if args.split == "all" else [args.split]

    manifest = json.loads((PROJECT_ROOT / "official_toolkit" / "config" / "corridors.json").read_text(encoding="utf-8"))
    panels = [p["corridor_id"] for p in manifest["panels"]]
    if args.panel:
        panels = [p for p in panels if p in set(args.panel)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    print(f"Generating Task 1 state reconstructions for {len(panels)} corridors across splits: {splits}")
    total_rows = 0
    first_header = True

    for panel in panels:
        count = build_panel_kf_submission(panel, root, splits, args.output.resolve(), first_header)
        total_rows += count
        first_header = False
        print(f"  [Summary {panel}]: wrote {count:,} rows")

    print(f"\nTask 1 complete! Successfully generated {total_rows:,} state reconstructions at {args.output}")


if __name__ == "__main__":
    main()
