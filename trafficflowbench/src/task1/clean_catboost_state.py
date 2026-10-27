"""Clean and enforce 100% FD bounds on CatBoost state predictions."""
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    root = Path("trafficflowbench")
    cat_path = root / "outputs" / "submissions" / "state_catboost.csv"
    out_path = root / "outputs" / "submissions" / "state_catboost_clean.csv"
    release = root / "data" / "kaggle_public"

    print("Loading CatBoost state predictions...")
    df = pd.read_csv(cat_path)
    print(f"Loaded {len(df):,} rows.")

    # Load link fd parameters for all corridors
    ffs_map = {}
    cap_map = {}
    corridors_dir = release / "corridors"
    for p in corridors_dir.iterdir():
        if p.is_dir():
            fd_file = p / "network" / "fd_parameters.csv"
            if fd_file.exists():
                fdf = pd.read_csv(fd_file, dtype={"link_id": str})
                for _, r in fdf.iterrows():
                    key = (p.name, str(r["link_id"]))
                    ffs_map[key] = float(r.get("free_speed_kmh", 105.0))
                    cap_map[key] = float(r.get("capacity_vph", 8000.0))

    panels = df["panel"].astype(str).to_numpy()
    links = df["link_id"].astype(str).to_numpy()
    pairs = list(zip(panels, links))

    ffs_arr = np.array([ffs_map.get(k, 105.0) for k in pairs], dtype=np.float64)
    cap_arr = np.array([cap_map.get(k, 8000.0) for k in pairs], dtype=np.float64)

    speed = df["speed_kmh"].to_numpy(dtype=np.float64)
    flow = df["flow_vph"].to_numpy(dtype=np.float64)

    # 1. Enforce speed bounds [5.0, ffs + 5.0]
    speed_clipped = np.clip(speed, 5.0, ffs_arr + 5.0)

    # 2. Enforce flow bounds [100.0, capacity]
    flow_clipped = np.clip(flow, 100.0, cap_arr)

    n_spd_changed = (speed != speed_clipped).sum()
    n_flw_changed = (flow != flow_clipped).sum()
    print(f"Speed values bounded: {n_spd_changed:,} / {len(df):,} ({100 * n_spd_changed / len(df):.4f}%)")
    print(f"Flow values bounded: {n_flw_changed:,} / {len(df):,} ({100 * n_flw_changed / len(df):.4f}%)")

    df["speed_kmh"] = speed_clipped
    df["flow_vph"] = flow_clipped

    print(f"Writing cleaned state to {out_path}...")
    df.to_csv(out_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
