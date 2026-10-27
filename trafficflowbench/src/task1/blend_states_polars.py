"""Key-aligned blend of CatBoost and 2D Spatiotemporal state predictions using Polars."""
from pathlib import Path
import json, time
import polars as pl
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
    out_path = SUB_DIR / "state_blended_fd_proper.csv"

    print("Loading CatBoost and 2D Spatiotemporal state tables with Polars...")
    df_cat = pl.read_csv(cat_path)
    df_spatio = pl.read_csv(spatio_path)

    keys = ["panel", "timestamp", "station_id", "link_id", "mask_regime"]
    print(f"CatBoost rows: {len(df_cat):,}, Spatio rows: {len(df_spatio):,}")

    # Inner join on keys
    print("Performing key-aligned inner join on", keys, "...")
    joined = df_cat.join(df_spatio, on=keys, suffix="_spatio")
    print(f"Joined rows: {len(joined):,}")
    assert len(joined) == 6735795, f"Expected 6,735,795 rows but got {len(joined):,}"

    # Calculate 50/50 blend
    joined = joined.with_columns([
        (0.5 * pl.col("speed_kmh") + 0.5 * pl.col("speed_kmh_spatio")).alias("speed_blend"),
        (0.5 * pl.col("flow_vph") + 0.5 * pl.col("flow_vph_spatio")).alias("flow_blend")
    ])

    print("Applying Fundamental Diagram physical constraints...")
    panel_cfgs = load_panel_configs()
    link_lanes = get_link_lanes()

    panels = joined["panel"].to_numpy()
    link_ids = joined["link_id"].to_numpy()
    speed_arr = joined["speed_blend"].to_numpy()
    flow_arr = joined["flow_blend"].to_numpy()

    pair_keys = list(zip(panels, link_ids))
    lanes_arr = np.array([link_lanes.get(k, 3.0) for k in pair_keys], dtype=np.float64)
    ffs_arr = np.array([float(panel_cfgs.get(p, {}).get("free_flow_speed", 105.0)) for p in panels], dtype=np.float64)
    cap_arr = np.array([float(panel_cfgs.get(p, {}).get("capacity", 2200.0)) for p in panels], dtype=np.float64) * lanes_arr
    jam_k_arr = np.array([float(panel_cfgs.get(p, {}).get("jam_density", 120.0)) for p in panels], dtype=np.float64)

    # Vectorized bounds
    # Flow: floor at 100 vph, cap at corridor capacity
    flow_arr = np.clip(flow_arr, 100.0, cap_arr)

    # Speed: min 5.0 km/h, max 1.15 * ffs
    speed_arr = np.clip(speed_arr, 5.0, ffs_arr * 1.15)

    # Density constraint: speed >= flow / (jam_k * lanes)
    min_speed_for_jam = flow_arr / (jam_k_arr * lanes_arr + 1e-6)
    speed_arr = np.maximum(speed_arr, min_speed_for_jam)

    print("FD bounds applied!")
    print(f"Final Speed: min={speed_arr.min():.1f}, max={speed_arr.max():.1f}, mean={speed_arr.mean():.1f}")
    print(f"Final Flow: min={flow_arr.min():.1f}, max={flow_arr.max():.1f}, mean={flow_arr.mean():.1f}")

    # Replace blended columns in polars frame
    out_df = joined.select(keys).with_columns([
        pl.Series("speed_kmh", speed_arr),
        pl.Series("flow_vph", flow_arr)
    ])

    print(f"Writing output to {out_path}...")
    out_df.write_csv(out_path)
    print(f"Done in {time.time()-t0:.1f}s!")

if __name__ == "__main__":
    main()
