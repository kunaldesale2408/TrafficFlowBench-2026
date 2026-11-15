import pandas as pd
import numpy as np
import sys
from pathlib import Path
import subprocess

ROOT = Path("trafficflowbench")
DATA = ROOT / "data" / "kaggle_public"
SEM_VAL = ROOT / "kaggle_research" / "semwal_087" / "output" / "task1" / "task1_validation.csv"
S24_ALL = ROOT / "outputs" / "submissions" / "state_s24_ensemble_fd.csv"

print("Loading Semwal validation state...")
sem_df = pd.read_csv(SEM_VAL)
print(f"Semwal val rows: {len(sem_df):,}")

# Filter to panel D12_I5_N first for an instant benchmark
panel = "D12_I5_N"
sem_p = sem_df[sem_df["panel"] == panel].copy()
print(f"Semwal {panel} rows: {len(sem_p):,}")

print(f"Loading S24 state for {panel} validation...")
# Read in chunks
keys = ["panel", "timestamp", "station_id", "link_id", "mask_regime"]
s24_chunks = []
for chunk in pd.read_csv(S24_ALL, chunksize=200_000):
    cp = chunk[(chunk["panel"] == panel) & (chunk["timestamp"].str[:7] == "2031-03")]
    if not cp.empty:
        s24_chunks.append(cp)

s24_p = pd.concat(s24_chunks, ignore_index=True)
print(f"S24 {panel} rows: {len(s24_p):,}")

# Align by keys
merged = sem_p.merge(s24_p, on=keys, suffixes=("_sem", "_s24"), how="inner")
print(f"Merged {panel} rows: {len(merged):,}")

# Evaluate:
# 1. Semwal pure
sem_out = merged[keys].copy()
sem_out["speed_kmh"] = merged["speed_kmh_sem"]
sem_out["flow_vph"] = merged["flow_vph_sem"]

# 2. S24 pure
s24_out = merged[keys].copy()
s24_out["speed_kmh"] = merged["speed_kmh_s24"]
s24_out["flow_vph"] = merged["flow_vph_s24"]

# 3. Champion 50/50 blend
blend_out = merged[keys].copy()
blend_out["speed_kmh"] = 0.50 * merged["speed_kmh_sem"] + 0.50 * merged["speed_kmh_s24"]
blend_out["flow_vph"] = 0.50 * merged["flow_vph_sem"] + 0.50 * merged["flow_vph_s24"]

def score_state(df, label):
    tmp = ROOT / "scratch" / f"tmp_state_{label}.csv"
    df.to_csv(tmp, index=False)
    cmd = [
        sys.executable,
        str(ROOT / "official_toolkit" / "src" / "task3" / "score_task3.py"),
        "--state-submission", str(tmp),
        "--release-root", str(DATA),
        "--split", "validation",
        "--panel", panel,
        "--output", str(ROOT / "scratch" / f"task3_eval_{label}.csv")
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    out_csv = ROOT / "scratch" / f"task3_eval_{label}.csv"
    if out_csv.exists():
        r_df = pd.read_csv(out_csv)
        print(f"\n--- {label.upper()} RESULTS ---")
        print(r_df[["panel", "regime", "S_FD", "S_LWR", "E_LWR", "S_physics"]])
    else:
        print(f"Failed {label}:", res.stderr)

score_state(sem_out, "semwal")
score_state(s24_out, "s24")
score_state(blend_out, "blend_50_50")
