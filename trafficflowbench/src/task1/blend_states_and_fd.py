"""Blend CatBoost and 2D Spatiotemporal state predictions with vectorized Fundamental Diagram compliance."""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd

DATA = Path("data/kaggle_public").resolve()
SUB_DIR = Path("outputs/submissions").resolve()

def load_panel_configs() -> dict:
    corridors_dir = DATA / "corridors"
    cfgs = {}
    for p in corridors_dir.iterdir():
        if p.is_dir():
            cfg_file = p / "config.json"
            if cfg_file.exists():
                cfgs[p.name] = json.loads(cfg_file.read_text(encoding="utf-8"))
    return cfgs

def get_link_lanes() -> dict:
    lanes_map = {}
    corridors_dir = DATA / "corridors"
    for p in corridors_dir.iterdir():
        if p.is_dir():
            links_file = p / "network" / "links.csv"
            if links_file.exists():
                ldf = pd.read_csv(links_file, dtype={"link_id": str})
                for _, r in ldf.iterrows():
                    lanes_map[(p.name, str(r["link_id"]))] = float(r.get("lanes", 3.0))
    return lanes_map

def main():
    t0 = time.time()
    cat_path = SUB_DIR / "state_catboost.csv"
    spatio_path = SUB_DIR / "state_submission.csv"
    out_path = SUB_DIR / "state_blended_fd.csv"

    print("Loading CatBoost state predictions...")
    cat_df = pd.read_csv(cat_path)
    print("Loading 2D Spatiotemporal state predictions...")
    spatio_df = pd.read_csv(spatio_path)

    assert len(cat_df) == len(spatio_df) == 6735795, f"Length mismatch: cat={len(cat_df)}, spatio={len(spatio_df)}"

    print("Blending predictions 50/50...")
    blend_speed = 0.5 * cat_df["speed_kmh"].to_numpy(dtype=np.float64) + 0.5 * spatio_df["speed_kmh"].to_numpy(dtype=np.float64)
    blend_flow = 0.5 * cat_df["flow_vph"].to_numpy(dtype=np.float64) + 0.5 * spatio_df["flow_vph"].to_numpy(dtype=np.float64)

    print("Vectorizing Fundamental Diagram bounds per panel...")
    panel_cfgs = load_panel_configs()
    link_lanes = get_link_lanes()

    panels = cat_df["panel"].to_numpy()
    link_ids = cat_df["link_id"].astype(str).to_numpy()

    # Pre-map lanes, ffs, capacity, jam_k arrays
    pair_keys = list(zip(panels, link_ids))
    lanes_arr = np.array([link_lanes.get(k, 3.0) for k in pair_keys], dtype=np.float64)
    ffs_arr = np.array([float(panel_cfgs.get(p, {}).get("free_flow_speed", 105.0)) for p in panels], dtype=np.float64)
    cap_arr = np.array([float(panel_cfgs.get(p, {}).get("capacity", 2200.0)) for p in panels], dtype=np.float64) * lanes_arr
    jam_k_arr = np.array([float(panel_cfgs.get(p, {}).get("jam_density", 120.0)) for p in panels], dtype=np.float64)

    # Vectorized clipping
    # 1. Flow bounds: [100.0, cap_arr]
    blend_flow = np.clip(blend_flow, 100.0, cap_arr)

    # 2. Speed bounds: min speed 5.0, max speed 1.15 * ffs
    blend_speed = np.clip(blend_speed, 5.0, ffs_arr * 1.15)

    # 3. Density constraint: k = flow / (speed * lanes) <= jam_k
    # => speed >= flow / (jam_k * lanes)
    min_speed_for_jam = blend_flow / (jam_k_arr * lanes_arr + 1e-6)
    blend_speed = np.maximum(blend_speed, min_speed_for_jam)

    print("FD bounds applied in vectorized form!")
    print(f"Blended Speed: min={blend_speed.min():.1f}, max={blend_speed.max():.1f}, mean={blend_speed.mean():.1f}")
    print(f"Blended Flow: min={blend_flow.min():.1f}, max={blend_flow.max():.1f}, mean={blend_flow.mean():.1f}")

    result = cat_df[["panel", "timestamp", "station_id", "link_id", "mask_regime"]].copy()
    result["speed_kmh"] = blend_speed
    result["flow_vph"] = blend_flow

    print(f"Writing {len(result):,} rows to {out_path}...")
    result.to_csv(out_path, index=False)
    print(f"Done in {time.time()-t0:.1f}s!")

if __name__ == "__main__":
    main()
