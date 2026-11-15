import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("trafficflowbench")
s44 = pd.read_csv(ROOT / "outputs" / "submissions" / "queue_s44_precision.csv")
probs = pd.read_parquet(ROOT / "scratch" / "test_ongoing_probabilities.parquet")

w10 = s44[s44['window_id'] == 'Q2_D7_I405_N_validation_010'].copy()
print(f"Total cells in Q2_D7_I405_N_validation_010: {len(w10)}")
print(f"Unique links: {w10['link_id'].nunique()}")
print(f"Unique timestamps: {w10['timestamp'].nunique()}")
print(f"Total positives: {w10['queue_pred'].sum()}")

# Check what the speed was during history (before T) for this window!
hist_path = ROOT / "data" / "kaggle_public" / "task2" / "D7_I405_N" / "validation" / "window_history.parquet"
hist = pd.read_parquet(hist_path)
w10_hist = hist[hist['window_id'] == 'Q2_D7_I405_N_validation_010']
print(f"\nHistory rows: {len(w10_hist)}")
print("History speed stats:")
print(w10_hist['speed_kmh'].describe())
print(f"History rows with speed < 50 km/h: {(w10_hist['speed_kmh'] < 50).sum()} / {len(w10_hist)}")
print(f"History rows with speed >= 100 km/h: {(w10_hist['speed_kmh'] >= 100).sum()} / {len(w10_hist)}")

# Check forecast origin and times
wi = pd.read_csv(ROOT / "data" / "kaggle_public" / "task2" / "D7_I405_N" / "validation" / "window_index.csv")
print("\nWindow index info:")
print(wi[wi['window_id'] == 'Q2_D7_I405_N_validation_010'])
