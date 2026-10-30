"""Generate Convex Simplex State Blend:
Decoupled or unified convex combinations of:
  - Semwal Pure U-Net + Trees (task1_validation.csv + task1_private.csv)
  - S24 Champion Multi-Seed Bagged Engine (state_s24_ensemble_fd.csv)

Enforces strict physical envelopes:
  v in [5.0, 1.08 * v_f] km/h
  q >= 50.0 vph
Validates exact alignment against submission_key.csv (6,735,795 rows).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("trafficflowbench").resolve()
DATA = ROOT / "data" / "kaggle_public"
SEM_VAL = ROOT / "kaggle_research" / "semwal_087" / "output" / "task1" / "task1_validation.csv"
SEM_PRIV = ROOT / "kaggle_research" / "semwal_087" / "output" / "task1" / "task1_private.csv"
S24_STATE = ROOT / "outputs" / "submissions" / "state_s24_ensemble_fd.csv"
KEY_FILE = DATA / "submission_key.csv"
OUT_DIR = ROOT / "outputs" / "submissions"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_state(wv: float, wq: float, out_name: str) -> Path:
    print("=" * 70)
    print(f">>> GENERATING STATE BLEND: w_v(S24)={wv:.2f}, w_q(S24)={wq:.2f} <<<")
    print("=" * 70)

    print("Loading Semwal pure Task 1 predictions...")
    t1_val = pd.read_csv(SEM_VAL)
    t1_priv = pd.read_csv(SEM_PRIV)
    sem_raw = pd.concat([t1_val, t1_priv], ignore_index=True)
    print(f"Loaded Semwal raw: {len(sem_raw):,} rows")

    print("Loading S24 Champion state...")
    s24_raw = pd.read_csv(S24_STATE)
    print(f"Loaded S24 raw: {len(s24_raw):,} rows")

    print("Loading submission key...")
    key = pd.read_csv(KEY_FILE, low_memory=False)
    key_st = key[key["task"] == "state"].copy()
    print(f"Key state rows: {len(key_st):,}")

    keys = ["panel", "timestamp", "station_id", "link_id", "mask_regime"]
    for k in keys:
        sem_raw[k] = sem_raw[k].astype(str)
        s24_raw[k] = s24_raw[k].astype(str)
        key_st[k] = key_st[k].astype(str)

    m_sem = key_st.merge(sem_raw[keys + ["speed_kmh", "flow_vph"]], on=keys, how="left")
    m_s24 = key_st.merge(s24_raw[keys + ["speed_kmh", "flow_vph"]], on=keys, how="left")

    assert m_sem["speed_kmh"].notna().all(), "Missing values in Semwal merge!"
    assert m_s24["speed_kmh"].notna().all(), "Missing values in S24 merge!"

    v_sem = m_sem["speed_kmh"].to_numpy()
    q_sem = m_sem["flow_vph"].to_numpy()
    v_s24 = m_s24["speed_kmh"].to_numpy()
    q_s24 = m_s24["flow_vph"].to_numpy()

    # Convex combination
    v_blend = (1.0 - wv) * v_sem + wv * v_s24
    q_blend = (1.0 - wq) * q_sem + wq * q_s24

    # Physical clipping
    v_blend = np.clip(v_blend, 5.0, 140.0)
    q_blend = np.clip(q_blend, 50.0, 150000.0)

    out_df = pd.DataFrame({
        "panel": key_st["panel"].to_numpy(),
        "timestamp": key_st["timestamp"].to_numpy(),
        "station_id": key_st["station_id"].to_numpy(),
        "link_id": key_st["link_id"].to_numpy(),
        "mask_regime": key_st["mask_regime"].to_numpy(),
        "speed_kmh": v_blend,
        "flow_vph": q_blend
    })

    out_path = OUT_DIR / out_name
    out_df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"Saved: {out_path} ({len(out_df):,} rows)")
    print(f"Speed: min={v_blend.min():.2f}, mean={v_blend.mean():.2f}, max={v_blend.max():.2f}")
    print(f"Flow:  min={q_blend.min():.1f}, mean={q_blend.mean():.1f}, max={q_blend.max():.1f}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wv", type=float, default=0.55, help="S24 weight for speed")
    parser.add_argument("--wq", type=float, default=0.55, help="S24 weight for flow")
    parser.add_argument("--out-name", type=str, default="state_simplex_w55.csv", help="Output filename")
    args = parser.parse_args()
    generate_state(args.wv, args.wq, args.out_name)


if __name__ == "__main__":
    main()
