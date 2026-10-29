import os
os.environ['TFB_DATA_ROOT'] = 'trafficflowbench/data/kaggle_public'
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path('trafficflowbench/data/kaggle_public')

ffs = {}
for p in (ROOT / 'corridors').iterdir():
    if p.is_dir():
        fd = p / 'network' / 'fd_parameters.csv'
        if fd.exists():
            df = pd.read_csv(fd, dtype={'link_id': str})
            for _, r in df.iterrows():
                ffs[(p.name, str(r['link_id']))] = float(r.get('free_speed_kmh', 105.0))

true_queue_counts = []
for p in (ROOT / 'corridors').iterdir():
    if not p.is_dir():
        continue
    name = p.name
    w_file = ROOT / 'task2' / name / 'train' / 'window_index.csv'
    if not w_file.exists():
        continue
    windows = pd.read_csv(w_file, dtype={'window_id': str})
    onset = windows[windows.condition == 'queue_onset']
    
    state_files = sorted((p / 'train' / 'mainline_states').glob('**/*.parquet'))
    if not state_files:
        continue
    s_dfs = [pd.read_parquet(sf, columns=['timestamp', 'link_id', 'speed_kmh']) for sf in state_files]
    state_df = pd.concat(s_dfs, ignore_index=True)
    state_df['timestamp'] = pd.to_datetime(state_df['timestamp'], utc=True)
    state_df['link_id'] = state_df['link_id'].astype(str)
    
    for row in onset.itertuples():
        t30 = pd.Timestamp(row.forecast_origin) + pd.Timedelta(minutes=30)
        snap = state_df[state_df.timestamp == t30]
        if snap.empty:
            continue
        q_links = []
        for _, sr in snap.iterrows():
            v_cut = 0.60 * ffs.get((name, sr['link_id']), 105.0)
            if sr['speed_kmh'] <= v_cut:
                q_links.append(sr['link_id'])
        true_queue_counts.append({'window_id': row.window_id, 'panel': name, 'n_true_queues': len(q_links), 'links': q_links})

tdf = pd.DataFrame(true_queue_counts)
print(f'Total train onset windows evaluated: {len(tdf)}')
print('Distribution of TRUE queue link counts at T+30:')
print(tdf['n_true_queues'].value_counts().sort_index())
print(f"Mean true queue links: {tdf['n_true_queues'].mean():.2f}")
print(f"Median true queue links: {tdf['n_true_queues'].median():.1f}")
print(f"Percentage with <= 1 true queue: {(tdf['n_true_queues'] <= 1).mean():.1%}")
print(f"Percentage with <= 2 true queues: {(tdf['n_true_queues'] <= 2).mean():.1%}")
print(f"Percentage with <= 3 true queues: {(tdf['n_true_queues'] <= 3).mean():.1%}")
