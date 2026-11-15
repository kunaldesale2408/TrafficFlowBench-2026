import pandas as pd
import numpy as np
from pathlib import Path
import json

ROOT = Path("trafficflowbench")
DATA = ROOT / "data" / "kaggle_public"
SUB_S48 = ROOT / "outputs" / "submissions" / "submission_s48.csv"

# Load S48
key = pd.read_csv(DATA / "submission_key.csv", low_memory=False)
sub = pd.read_csv(SUB_S48)

st_mask = key["task"] == "state"
key_st = key[st_mask].copy()
key_st["speed_kmh"] = sub.loc[st_mask, "speed_kmh"].values
key_st["flow_vph"] = sub.loc[st_mask, "flow_vph"].values

# Filter to validation
val_st = key_st[key_st["timestamp"].str[:7] == "2031-03"].copy()

# Load network parameters per panel
corridors = sorted([p.name for p in (DATA / "corridors").iterdir() if p.is_dir()])

# Compute observed medians for validation
obs_meds = {}
for p in corridors:
    msk_files = sorted((DATA / "corridors" / p / "validation" / "mainline_states_masked").glob("**/*.parquet"))
    dfs = [pd.read_parquet(f, columns=['link_id', 'speed_kmh']).dropna() for f in msk_files]
    if dfs:
        df_sp = pd.concat(dfs, ignore_index=True)
        meds = df_sp[df_sp['speed_kmh'] >= 90.0].groupby('link_id')['speed_kmh'].median().to_dict()
        for l, val in meds.items():
            obs_meds[(p, str(l))] = float(val)

val_st["target_ff_med"] = [obs_meds.get((p, str(l)), 115.0) for p, l in zip(val_st["panel"], val_st["link_id"])]

# Function to compute S_FD
def compute_s_fd(df):
    panel_sfds = []
    for p in corridors:
        p_df = df[df["panel"] == p].copy()
        fd_path = DATA / "corridors" / p / "network" / "fd_parameters.csv"
        links_path = DATA / "corridors" / p / "network" / "links.csv"
        
        links = pd.read_csv(links_path, dtype={"link_id": str})
        links = links.drop_duplicates("link_id").set_index("link_id")
        
        fdf = pd.read_csv(fd_path, dtype={"link_id": str})
        fdf = fdf.groupby("link_id").mean(numeric_only=True)
        
        p_df = p_df.join(links[["lanes"]], on="link_id")
        p_df = p_df.join(fdf[["free_speed_kmh", "capacity_vph", "k_jam"]], on="link_id", rsuffix="_fd")
        
        lanes = p_df["lanes"].fillna(4.0).clip(lower=1.0)
        vf = p_df["free_speed_kmh"].fillna(105.0)
        cap = p_df["capacity_vph"].fillna(1800.0 * lanes)
        kcrit = cap / vf.clip(lower=1.0)
        kjam = p_df["k_jam"].fillna(100.0 * lanes).clip(lower=kcrit * 1.05)
        
        q_lane = p_df["flow_vph"] / lanes
        k_lane = (p_df["flow_vph"] / p_df["speed_kmh"].clip(lower=1.0)) / lanes
        cap_lane = cap / lanes
        kcrit_lane = kcrit / lanes
        kjam_lane = kjam / lanes
        
        q_fd = np.where(
            k_lane <= kcrit_lane,
            vf * k_lane,
            (cap_lane / np.maximum(kjam_lane - kcrit_lane, 1e-9)) * np.maximum(kjam_lane - k_lane, 0.0)
        )
        s_fd = float(max(0.0, 1.0 - np.abs(q_lane - q_fd).sum() / (np.abs(q_lane).sum() + 1e-9)))
        panel_sfds.append(s_fd)
    return float(np.mean(panel_sfds))

base_sfd = compute_s_fd(val_st)
print(f"Baseline S48 S_FD on validation: {base_sfd:.6f}")

for tau in [100.0, 105.0, 108.0]:
    for alpha in [0.05, 0.10, 0.15, 0.20]:
        cand = val_st.copy()
        gate = cand["speed_kmh"] >= tau
        cand.loc[gate, "speed_kmh"] = (1 - alpha) * cand.loc[gate, "speed_kmh"] + alpha * cand.loc[gate, "target_ff_med"]
        cand["speed_kmh"] = np.clip(cand["speed_kmh"], 6.0, 120.75)
        s_fd = compute_s_fd(cand)
        print(f"tau={tau:.1f}, alpha={alpha:.2f} -> S_FD = {s_fd:.6f} (Delta: {s_fd - base_sfd:+.6f})")
