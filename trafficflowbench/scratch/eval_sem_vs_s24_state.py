import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("trafficflowbench")
sem_val = pd.read_csv(ROOT / "kaggle_research" / "semwal_087" / "output" / "task1" / "task1_validation.csv")
s24_val = pd.read_csv(ROOT / "outputs" / "submissions" / "state_s24_ensemble_fd.csv")
s24_val = s24_val[s24_val['timestamp'].str[:7] == '2031-03'].copy()

keys = ["panel", "timestamp", "station_id", "link_id", "mask_regime"]
for k in keys:
    sem_val[k] = sem_val[k].astype(str)
    s24_val[k] = s24_val[k].astype(str)

m = sem_val[keys + ['speed_kmh', 'flow_vph']].merge(
    s24_val[keys + ['speed_kmh', 'flow_vph']],
    on=keys,
    how='inner',
    suffixes=('_sem', '_s24')
)

print(f"Matched rows: {len(m):,}")
print("\nSpeed summary:")
print(f"  Semwal: min={m['speed_kmh_sem'].min():.2f}, mean={m['speed_kmh_sem'].mean():.2f}, max={m['speed_kmh_sem'].max():.2f}")
print(f"  S24   : min={m['speed_kmh_s24'].min():.2f}, mean={m['speed_kmh_s24'].mean():.2f}, max={m['speed_kmh_s24'].max():.2f}")
print(f"  Correlation: {m['speed_kmh_sem'].corr(m['speed_kmh_s24']):.5f}")
print(f"  RMSE diff  : {np.sqrt(((m['speed_kmh_sem'] - m['speed_kmh_s24'])**2).mean()):.3f} km/h")

print("\nFlow summary:")
print(f"  Semwal: min={m['flow_vph_sem'].min():.2f}, mean={m['flow_vph_sem'].mean():.2f}, max={m['flow_vph_sem'].max():.2f}")
print(f"  S24   : min={m['flow_vph_s24'].min():.2f}, mean={m['flow_vph_s24'].mean():.2f}, max={m['flow_vph_s24'].max():.2f}")
print(f"  Correlation: {m['flow_vph_sem'].corr(m['flow_vph_s24']):.5f}")
print(f"  RMSE diff  : {np.sqrt(((m['flow_vph_sem'] - m['flow_vph_s24'])**2).mean()):.3f} vph")
print(f"  Mean diff  : {(m['flow_vph_s24'] - m['flow_vph_sem']).mean():.2f} vph")
