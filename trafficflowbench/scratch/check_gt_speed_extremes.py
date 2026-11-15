import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("trafficflowbench")
DATA = ROOT / "data" / "kaggle_public"

mins = []
maxs = []
for p in (DATA / "corridors").iterdir():
    if not p.is_dir(): continue
    for f in (p / "train" / "mainline_states").glob("**/*.parquet"):
        df = pd.read_parquet(f, columns=['speed_kmh', 'flow_vph'])
        s = df['speed_kmh'].dropna()
        if len(s):
            mins.append(s.min())
            maxs.append(s.max())

print(f"Scanned {len(mins)} day files across all corridors.")
print(f"Overall min speed in ground truth: {min(mins):.4f} km/h")
print(f"Overall max speed in ground truth: {max(maxs):.4f} km/h")
