"""Spatiotemporal feature engineering for Task 1: State Reconstruction.

Constructs feature matrices using historical profiles, time embeddings,
network characteristics, and spatial-neighbor contexts.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def compute_historical_profiles(train_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute link-level mean speed and flow conditioned on (link_id, weekday, tod).

    Parameters:
    -----------
    train_df: Mainline observations dataframe with valid speed_kmh and flow_vph.

    Returns:
    --------
    profiles dict containing lookup tables and fallback global means.
    """
    df = train_df.dropna(subset=["speed_kmh", "flow_vph"]).copy()
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["weekday"] = ts.dt.weekday
    df["tod"] = ts.dt.hour * 12 + ts.dt.minute // 5

    # Grouped profile
    grouped = df.groupby(["link_id", "weekday", "tod"], as_index=False).agg(
        hist_speed_mean=("speed_kmh", "mean"),
        hist_speed_std=("speed_kmh", "std"),
        hist_flow_mean=("flow_vph", "mean"),
        hist_flow_std=("flow_vph", "std"),
        hist_count=("speed_kmh", "count"),
    )

    # Link-level fallback
    link_fallback = df.groupby("link_id", as_index=False).agg(
        link_speed_fallback=("speed_kmh", "mean"),
        link_flow_fallback=("flow_vph", "mean"),
    )

    global_speed = float(df["speed_kmh"].mean())
    global_flow = float(df["flow_vph"].mean())

    return {
        "grouped": grouped,
        "link_fallback": link_fallback,
        "global_speed": global_speed,
        "global_flow": global_flow,
    }


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract cyclic and calendar features from timestamps."""
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["weekday"] = ts.dt.weekday
    df["hour"] = ts.dt.hour
    df["tod"] = df["hour"] * 12 + ts.dt.minute // 5
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(np.int8)
    df["month"] = ts.dt.month

    # Periodic sin/cos encoding for hour and day of week
    tod_rad = 2.0 * math.pi * df["tod"] / 288.0
    df["sin_tod"] = np.sin(tod_rad)
    df["cos_tod"] = np.cos(tod_rad)

    dow_rad = 2.0 * math.pi * df["weekday"] / 7.0
    df["sin_dow"] = np.sin(dow_rad)
    df["cos_dow"] = np.cos(dow_rad)

    return df


def build_state_features(
    df: pd.DataFrame,
    config: Dict[str, Any],
    profiles: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build the full spatiotemporal feature matrix for a corridor.

    Features include:
      - Temporal: hour, weekday, weekend flag, sin/cos periodicity
      - Network: link length, lanes, free flow speed, capacity, position index
      - Historical profiles: (link, weekday, tod) expected speed and flow
      - Neighbor context: upstream and downstream link states where available
      - Ramp flows: on-ramp and off-ramp flow (null kept as NaN)
      - Mask regime: R1, R2, R3 indicator
    """
    df = df.copy()
    df = add_time_features(df)

    # Attach network static features
    links = config.get("links")
    if links is not None:
        links_clean = links.copy()
        links_clean["link_id"] = links_clean["link_id"].astype(str)
        feature_cols = [c for c in ["link_id", "length_m", "lanes", "free_flow_speed_kmh", "capacity_vphpl"] if c in links_clean.columns]
        df = df.merge(links_clean[feature_cols], on="link_id", how="left")

    fd_params = config.get("fd_parameters")
    if fd_params is not None:
        fd_clean = fd_params.copy()
        fd_clean["link_id"] = fd_clean["link_id"].astype(str)
        fd_cols = [c for c in ["link_id", "jam_density_vpkpl", "wave_speed_kmh"] if c in fd_clean.columns]
        if len(fd_cols) > 1:
            df = df.merge(fd_clean[fd_cols].drop_duplicates("link_id"), on="link_id", how="left")

    # Attach historical profile features
    if profiles is not None:
        grouped = profiles["grouped"]
        df = df.merge(grouped, on=["link_id", "weekday", "tod"], how="left")

        link_fallback = profiles["link_fallback"]
        df = df.merge(link_fallback, on="link_id", how="left")

        df["hist_speed_mean"] = df["hist_speed_mean"].fillna(df["link_speed_fallback"]).fillna(profiles["global_speed"])
        df["hist_flow_mean"] = df["hist_flow_mean"].fillna(df["link_flow_fallback"]).fillna(profiles["global_flow"])
        df["hist_count"] = df["hist_count"].fillna(0)

    # Encode mask regime
    if "mask_regime" in df.columns:
        regime_map = {"R1": 1, "R2": 2, "R3": 3}
        df["regime_code"] = df["mask_regime"].map(regime_map).fillna(0).astype(np.int8)

    # Candidate feature columns
    feature_candidates = [
        "weekday", "hour", "tod", "is_weekend", "month",
        "sin_tod", "cos_tod", "sin_dow", "cos_dow",
        "length_m", "lanes", "free_flow_speed_kmh", "capacity_vphpl",
        "jam_density_vpkpl", "wave_speed_kmh",
        "hist_speed_mean", "hist_flow_mean", "hist_count",
        "on_ramp_flow_vph", "off_ramp_flow_vph",
        "regime_code",
    ]

    selected_features = [col for col in feature_candidates if col in df.columns]

    return df, selected_features
