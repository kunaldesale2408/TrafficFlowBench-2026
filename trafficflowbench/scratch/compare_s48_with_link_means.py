import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("trafficflowbench")
DATA = ROOT / "data" / "kaggle_public"

# 1. Compute empirical link mean free-flow speed for all corridors and all links from train
print("Computing empirical link free-flow speed profile from train...")
link_ff_means = {}

corridors = sorted([p.name for p in (DATA / "corridors").iterdir() if p.is_dir()])
for p in corridors:
    unm_files = sorted((DATA / "corridors" / p / "train" / "mainline_states").glob("**/*.parquet"))
    # Load last 60 days of train (Jan-Feb 2031)
    dfs = []
    for f in unm_files[-60:]:
        dfs.append(pd.read_parquet(f, columns=['link_id', 'speed_kmh']))
    df = pd.concat(dfs, ignore_index=True)
    ff = df[df['speed_kmh'] >= 90.0]
    m = ff.groupby('link_id')['speed_kmh'].median().to_dict()
    for l, val in m.items():
        link_ff_means[(p, str(l))] = float(val)

print(f"Computed free-flow medians for {len(link_ff_means):,} corridor-link pairs.")

# 2. Check S48 speed predictions on validation
s48 = pd.read_csv(ROOT / "outputs" / "submissions" / "submission_s48.csv")
key = pd.read_csv(DATA / "submission_key.csv")
st_mask = key["task"] == "state"
s48_st = key[st_mask].copy()
s48_st["speed_kmh"] = s48.loc[st_mask, "speed_kmh"].values
s48_st["flow_vph"] = s48.loc[st_mask, "flow_vph"].values

# Filter to validation
val_st = s48_st[s48_st["timestamp"].str[:7] == "2031-03"].copy()
print(f"Validation state cells: {len(val_st):,}")

# Match each cell to its link free-flow median
val_st["ff_median"] = [link_ff_means.get((p, str(l)), np.nan) for p, l in zip(val_st["panel"], val_st["link_id"])]

# Look at free flow cells (speed >= 90 km/h in prediction)
val_ff = val_st[val_st["speed_kmh"] >= 90.0]
print(f"Free flow cells (pred speed >= 90): {len(val_ff):,} ({len(val_ff)/len(val_st):.2%})")

diff = val_ff["speed_kmh"] - val_ff["ff_median"]
print("\nDifference between S48 prediction and link empirical FF median:")
print(f"  Mean diff: {diff.mean():.4f} km/h")
print(f"  Std diff : {diff.std():.4f} km/h")
print(f"  RMSE diff: {np.sqrt((diff**2).mean()):.4f} km/h")
print(f"  Max diff : {diff.abs().max():.4f} km/h")
