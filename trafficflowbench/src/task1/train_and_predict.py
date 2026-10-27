"""Task 1 — Spatiotemporal LightGBM Training & Prediction Pipeline.

Trains per-corridor gradient-boosted trees with MAE loss for:
  1. speed_kmh
  2. flow_vph

Applies post-processing:
  - Fundamental diagram parameter bounds clipping
  - Flow-balance conservation smoothing
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task1.features import build_state_features, compute_historical_profiles
from src.task1.postprocess import enforce_fundamental_diagram_bounds, flow_balance_projection
from src.utils.data_loader import load_corridor, load_corridor_config, get_corridor_list


OUTPUT_COLUMNS = ["panel", "timestamp", "station_id", "link_id", "mask_regime", "speed_kmh", "flow_vph"]

LGB_PARAMS_SPEED = {
    "objective": "regression_l1",
    "n_estimators": 150,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_jobs": -1,
    "verbose": -1,
    "random_state": 42,
}

LGB_PARAMS_FLOW = {
    "objective": "regression_l1",
    "n_estimators": 150,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_jobs": -1,
    "verbose": -1,
    "random_state": 42,
}


def load_task1_template(release_root: Path, panel: str, split: str) -> pd.DataFrame:
    """Read the official sample_submission_state.csv template for this corridor."""
    template_path = release_root / "task1" / panel / split / "sample_submission_state.csv"
    if not template_path.exists():
        raise FileNotFoundError(f"Missing Task 1 template at: {template_path}")
    tmpl = pd.read_csv(template_path, dtype=str)
    tmpl["station_id"] = tmpl["station_id"].astype(str)
    tmpl["link_id"] = tmpl["link_id"].astype(str)
    return tmpl


def train_and_predict_panel(
    release_root: Path,
    panel: str,
    split: str = "validation",
    model_dir: Optional[Path] = None,
    max_train_files: Optional[int] = None,
) -> pd.DataFrame:
    """Train per-corridor speed/flow models and predict on target split."""
    print(f"\n[Task 1] Processing corridor: {panel} (target split: {split})")
    config = load_corridor_config(release_root, panel)

    # Check if pre-trained models exist
    speed_model_path = model_dir / f"lgb_speed_{panel}.joblib" if model_dir else None
    flow_model_path = model_dir / f"lgb_flow_{panel}.joblib" if model_dir else None
    profiles_path = model_dir / f"profiles_{panel}.joblib" if model_dir else None

    if (
        speed_model_path and speed_model_path.exists()
        and flow_model_path and flow_model_path.exists()
        and profiles_path and profiles_path.exists()
    ):
        print(f"  Found cached models for {panel} in {model_dir}. Loading...")
        speed_model = joblib.load(speed_model_path)
        flow_model = joblib.load(flow_model_path)
        profiles = joblib.load(profiles_path)
    else:
        # 1. Load train split
        print(f"  Loading train split for {panel}...")
        train_df, _ = load_corridor(
            release_root,
            panel,
            split="train",
            masked=False,
            include_ramps=True,
            sample_n_files=max_train_files,
        )

        # Compute historical profile tables from train observations
        print(f"  Computing historical link/time profiles...")
        profiles = compute_historical_profiles(train_df)

        # Build features on train
        train_df, feature_cols = build_state_features(train_df, config, profiles=profiles)
        print(f"  Constructed {len(feature_cols)} features: {feature_cols}")

        # Prepare training targets (only valid observed cells)
        valid_speed = train_df["speed_kmh"].notna()
        X_speed = train_df.loc[valid_speed, feature_cols]
        y_speed = train_df.loc[valid_speed, "speed_kmh"].to_numpy()

        valid_flow = train_df["flow_vph"].notna()
        X_flow = train_df.loc[valid_flow, feature_cols]
        y_flow = train_df.loc[valid_flow, "flow_vph"].to_numpy()

        # Train LightGBM Speed Model
        print(f"  Training Speed Model ({len(X_speed):,} samples)...")
        speed_model = lgb.LGBMRegressor(**LGB_PARAMS_SPEED)
        speed_model.fit(X_speed, y_speed)

        # Train LightGBM Flow Model
        print(f"  Training Flow Model ({len(X_flow):,} samples)...")
        flow_model = lgb.LGBMRegressor(**LGB_PARAMS_FLOW)
        flow_model.fit(X_flow, y_flow)

        # Save models
        if model_dir:
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(speed_model, speed_model_path)
            joblib.dump(flow_model, flow_model_path)
            joblib.dump(profiles, profiles_path)

    # 2. Load target split observations
    print(f"  Loading target split ({split}) data...")
    target_df, _ = load_corridor(
        release_root,
        panel,
        split=split,
        masked=True,
        include_ramps=True,
    )

    # Read official template to get exact target cells
    template = load_task1_template(release_root, panel, split)
    template_keys = set(zip(
        template["timestamp"].astype(str),
        template["station_id"].astype(str),
        template["link_id"].astype(str),
    ))

    # Build features on target data
    target_df, feature_cols = build_state_features(target_df, config, profiles=profiles)
    if hasattr(speed_model, "feature_name_") and speed_model.feature_name_:
        feature_cols = [c for c in speed_model.feature_name_ if c in target_df.columns]

    # Filter to template rows
    target_df["_key"] = list(zip(
        target_df["timestamp"].astype(str),
        target_df["station_id"].astype(str),
        target_df["link_id"].astype(str),
    ))
    eval_mask = target_df["_key"].isin(template_keys)
    predict_df = target_df.loc[eval_mask].copy()

    # Predict speed and flow
    print(f"  Generating predictions for {len(predict_df):,} target cells...")
    X_pred = predict_df[feature_cols]
    predict_df["speed_kmh"] = speed_model.predict(X_pred)
    predict_df["flow_vph"] = flow_model.predict(X_pred)

    # 3. Apply post-processing
    print("  Applying FD clipping and physics enforcement...")
    predict_df = enforce_fundamental_diagram_bounds(predict_df, config)

    print("  Applying flow-balance projection...")
    predict_df = flow_balance_projection(predict_df, config, strength=0.2)

    predict_df["panel"] = panel
    out = predict_df[OUTPUT_COLUMNS].copy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 1 State Reconstruction Pipeline")
    parser.add_argument("--release-root", type=Path, default=Path("./data/kaggle_public"))
    parser.add_argument("--split", choices=["train", "validation", "private", "all"], default="all")
    parser.add_argument("--output", type=Path, default=Path("outputs/submissions/state_submission.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/models"))
    parser.add_argument("--panel", action="append", help="Specific panel(s) to run")
    parser.add_argument("--max-train-files", type=int, default=30, help="Limit train parquet files for fast runs (default 30)")
    args = parser.parse_args()

    root = args.release_root.resolve()
    corridors = get_corridor_list(root)
    if args.panel:
        corridors = [p for p in corridors if p in args.panel]

    splits = ["validation", "private"] if args.split == "all" else [args.split]
    print(f"Starting Task 1 for {len(corridors)} corridors on split(s): {splits}")
    results = []
    for sp in splits:
        for panel in corridors:
            panel_df = train_and_predict_panel(
                root,
                panel,
                split=sp,
                model_dir=args.model_dir,
                max_train_files=args.max_train_files,
            )
            results.append(panel_df)
            print(f"  Corridor {panel} ({sp}) complete: {len(panel_df):,} rows.")

    combined = pd.concat(results, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"\nTask 1 successfully finished! Saved {len(combined):,} rows to {args.output}")


if __name__ == "__main__":
    main()
