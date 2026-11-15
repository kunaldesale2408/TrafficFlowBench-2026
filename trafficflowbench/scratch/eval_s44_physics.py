import pandas as pd
from pathlib import Path
import subprocess

ROOT = Path("trafficflowbench")
DATA = ROOT / "data" / "kaggle_public"
SUB = ROOT / "outputs" / "submissions" / "submission_s44.csv"
KEY = DATA / "submission_key.csv"

print("Loading submission_key.csv...")
key = pd.read_csv(KEY, low_memory=False)
key_st = key[key["task"] == "state"].copy()

print("Loading submission_s44.csv state rows...")
sub = pd.read_csv(SUB)
st_rows = sub[sub["task"] == "state"][["submission_id", "speed_kmh", "flow_vph"]]

merged = key_st.merge(st_rows, on="submission_id", how="inner")
print(f"Merged state rows: {len(merged):,}")

# Filter to validation
val_st = merged[merged["timestamp"].str[:7] == "2031-03"].copy()
print(f"Validation state rows: {len(val_st):,}")

temp_path = ROOT / "scratch" / "s44_val_state_for_task3.csv"
val_st.to_csv(temp_path, index=False)
print(f"Saved temporary validation state to {temp_path}")

cmd = [
    "python",
    str(ROOT / "official_toolkit" / "src" / "task3" / "score_task3.py"),
    "--state-submission", str(temp_path),
    "--release-root", str(DATA),
    "--split", "validation",
    "--panel", "D12_I5_N"
]

print("Running score_task3.py...")
res = subprocess.run(cmd, capture_output=True, text=True)
print("STDOUT:\n", res.stdout)
print("STDERR:\n", res.stderr)
