import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("trafficflowbench")
DATA = ROOT / "data" / "kaggle_public"
SUB_S48 = ROOT / "outputs" / "submissions" / "submission_s48.csv"

# 1. Load submission key & S48
key = pd.read_csv(DATA / "submission_key.csv", low_memory=False)
sub = pd.read_csv(SUB_S48)

st_mask = key["task"] == "state"
key_st = key[st_mask].copy()
key_st["speed_orig"] = sub.loc[st_mask, "speed_kmh"].values
key_st["flow_orig"] = sub.loc[st_mask, "flow_vph"].values

print(f"Total state cells: {len(key_st):,}")

# 2. Compute empirical link free-flow medians from train
corridors = sorted([p.name for p in (DATA / "corridors").iterdir() if p.is_dir()])
link_ff = {}
for p in corridors:
    unm_files = sorted((DATA / "corridors" / p / "train" / "mainline_states").glob("**/*.parquet"))
    dfs = [pd.read_parquet(f, columns=['link_id', 'speed_kmh']) for f in unm_files[-60:]]
    df_p = pd.concat(dfs, ignore_index=True)
    meds = df_p[df_p['speed_kmh'] >= 90.0].groupby('link_id')['speed_kmh'].median().to_dict()
    for l, val in meds.items():
        link_ff[(p, str(l))] = float(val)

print(f"Computed free-flow medians for {len(link_ff):,} corridor-link pairs.")

# 3. Also compute link medians from observed cells in validation and private
obs_meds = {}
for split in ["validation", "private"]:
    for p in corridors:
        msk_files = sorted((DATA / "corridors" / p / split / "mainline_states_masked").glob("**/*.parquet"))
        dfs = [pd.read_parquet(f, columns=['link_id', 'speed_kmh']).dropna() for f in msk_files]
        if dfs:
            df_sp = pd.concat(dfs, ignore_index=True)
            meds = df_sp[df_sp['speed_kmh'] >= 90.0].groupby('link_id')['speed_kmh'].median().to_dict()
            for l, val in meds.items():
                obs_meds[(split, p, str(l))] = float(val)

print(f"Computed observed medians for {len(obs_meds):,} split-corridor-link pairs.")

# Map target medians
key_st["split"] = np.where(key_st["timestamp"].str[:7] == "2031-03", "validation", "private")
med_list = []
for split, p, l in zip(key_st["split"], key_st["panel"], key_st["link_id"].astype(str)):
    val = obs_meds.get((split, p, l), link_ff.get((p, l), 115.0))
    med_list.append(val)
key_st["target_ff_med"] = med_list

# Test different gating thresholds and alphas
print("\n" + "="*70)
print(f"{'tau_free':<10} | {'alpha':<8} | {'Gated Cells':<14} | {'Mean Shift':<12} | {'Max Shift':<12}")
print("-" * 70)

for tau in [100.0, 105.0, 108.0, 110.0]:
    for alpha in [0.05, 0.10, 0.20, 0.30, 0.50]:
        gate = key_st["speed_orig"] >= tau
        v_cand = key_st["speed_orig"].copy()
        v_cand[gate] = (1 - alpha) * key_st.loc[gate, "speed_orig"] + alpha * key_st.loc[gate, "target_ff_med"]
        v_cand = np.clip(v_cand, 6.0, 120.75)
        shift = v_cand - key_st["speed_orig"]
        print(f"{tau:<10.1f} | {alpha:<8.2f} | {gate.sum():<14,} | {shift[gate].mean():<+12.4f} | {shift.abs().max():<12.4f}")
