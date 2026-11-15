import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("trafficflowbench")
DATA = ROOT / "data" / "kaggle_public"
q_df = pd.read_csv(ROOT / "outputs" / "submissions" / "queue_s44_precision.csv")

wi_files = list((DATA / "task2").glob("*/*/window_index.csv"))
wi_df = pd.concat([pd.read_csv(f) for f in wi_files], ignore_index=True)

cond_map = dict(zip(wi_df["window_id"].astype(str), wi_df["condition"].astype(str)))
origin_map = dict(zip(wi_df["window_id"].astype(str), wi_df["forecast_origin"].astype(str)))
panel_map = dict(zip(wi_df["window_id"].astype(str), wi_df["panel"].astype(str)))

q_df["condition"] = q_df["window_id"].astype(str).map(cond_map)
q_df["origin"] = q_df["window_id"].astype(str).map(origin_map)
q_df["panel"] = q_df["window_id"].astype(str).map(panel_map)
q_df["h"] = ((pd.to_datetime(q_df["timestamp"], utc=True) - pd.to_datetime(q_df["origin"], utc=True)).dt.total_seconds() // 300).astype(int)

# Load network free speed parameters
ffs_map = {}
for p in (DATA / "corridors").iterdir():
    if p.is_dir():
        fd_file = p / "network" / "fd_parameters.csv"
        links_file = p / "network" / "links.csv"
        if fd_file.exists():
            fdf = pd.read_csv(fd_file, dtype={"link_id": str})
            for _, r in fdf.iterrows():
                ffs_map[(p.name, str(r["link_id"]))] = float(r.get("free_speed_kmh", 105.0))
        elif links_file.exists():
            ldf = pd.read_csv(links_file, dtype={"link_id": str})
            for _, r in ldf.iterrows():
                ffs_map[(p.name, str(r["link_id"]))] = float(r.get("free_speed_kmh", 105.0))

print("FFS map loaded:", len(ffs_map))

# For each panel and split, read mainline_states_masked
obs_records = []
for split in ["validation", "private"]:
    for panel in wi_df["panel"].unique():
        masked_dir = DATA / "corridors" / panel / split / "mainline_states_masked"
        if not masked_dir.exists():
            continue
        for pfile in masked_dir.glob("**/*.parquet"):
            df = pd.read_parquet(pfile, columns=["timestamp", "link_id", "speed_kmh", "is_score_eligible"])
            df = df[df["is_score_eligible"].astype(bool) & df["speed_kmh"].notna()]
            if df.empty:
                continue
            df["panel"] = panel
            df["split"] = split
            obs_records.append(df[["panel", "timestamp", "link_id", "speed_kmh"]])

obs_all = pd.concat(obs_records, ignore_index=True)
obs_all["timestamp"] = pd.to_datetime(obs_all["timestamp"], utc=True)
obs_all["link_id"] = obs_all["link_id"].astype(str)
print(f"Total observed mainline state records loaded: {len(obs_all):,}")

# Filter ongoing h=1
ong_h1 = q_df[(q_df["condition"] == "queue_ongoing") & (q_df["h"] == 1)].copy()
ong_h1["origin_dt"] = pd.to_datetime(ong_h1["origin"], utc=True)
ong_h1["link_id"] = ong_h1["link_id"].astype(str)

merged = ong_h1.merge(
    obs_all.rename(columns={"timestamp": "origin_dt", "speed_kmh": "v_T"}),
    on=["panel", "origin_dt", "link_id"],
    how="inner"
)

print(f"Ongoing h=1 cells with direct observation at T: {len(merged):,} / {len(ong_h1):,} ({len(merged)/len(ong_h1)*100:.1f}%)")

merged["vf"] = [ffs_map.get((p, l), 105.0) for p, l in zip(merged["panel"], merged["link_id"])]
merged["v_ratio"] = merged["v_T"] / merged["vf"]

# Congestion threshold is v <= 0.60 * vf
merged["is_congested_at_T"] = merged["v_ratio"] <= 0.60
merged["is_clear_at_T"] = merged["v_ratio"] >= 0.85

print("\n--- CONGESTION AT T vs S44 PREDICTION AT h=1 ---")
print("Observed Congested at T (v_T <= 0.60 vf):", merged["is_congested_at_T"].sum())
print("  Of which S44 predicted 1:", (merged["is_congested_at_T"] & (merged["queue_pred"] == 1)).sum())
print("  Of which S44 predicted 0 (potential FN!):", (merged["is_congested_at_T"] & (merged["queue_pred"] == 0)).sum())

print("\nObserved Clear at T (v_T >= 0.85 vf):", merged["is_clear_at_T"].sum())
print("  Of which S44 predicted 0:", (merged["is_clear_at_T"] & (merged["queue_pred"] == 0)).sum())
print("  Of which S44 predicted 1 (potential FP!):", (merged["is_clear_at_T"] & (merged["queue_pred"] == 1)).sum())
