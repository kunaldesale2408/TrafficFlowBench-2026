import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("trafficflowbench")
DATA = ROOT / "data" / "kaggle_public"

corridors = sorted([p.name for p in (DATA / "corridors").iterdir() if p.is_dir()])
print(f"Corridors ({len(corridors)}): {corridors}")

# Let's inspect D12_I5_N across multiple train months
p = "D12_I5_N"
unm_files = sorted((DATA / "corridors" / p / "train" / "mainline_states").glob("**/*.parquet"))
print(f"\nAnalyzing corridor {p} across {len(unm_files)} days...")

# Sample 30 days
dfs = []
for f in unm_files[-30:]:
    d = pd.read_parquet(f, columns=['timestamp', 'link_id', 'speed_kmh', 'flow_vph'])
    dfs.append(d)
df = pd.concat(dfs, ignore_index=True)

# Free flow: speed >= 90 km/h
ff = df[df['speed_kmh'] >= 90.0]
link_stats = ff.groupby('link_id')['speed_kmh'].agg(['count', 'mean', 'std', 'min', 'max'])
print("\nLink free-flow speed stats (sample of 15 links):")
print(link_stats.head(15).to_string())

print(f"\nOverall average between-day/within-link std in free flow: {link_stats['std'].mean():.3f} km/h")
print(f"Max observed speed across sample: {link_stats['max'].max():.4f} km/h")
