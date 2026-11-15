import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("trafficflowbench")
DATA = ROOT / "data" / "kaggle_public"

s44 = pd.read_csv(ROOT / "outputs" / "submissions" / "queue_s44_precision.csv")
probs = pd.read_parquet(ROOT / "scratch" / "test_ongoing_probabilities.parquet")

wi_df = pd.concat([pd.read_csv(f) for f in (pd.Series(list(pd.io.common.Path('trafficflowbench/data/kaggle_public/task2').glob('*/*/window_index.csv'))))], ignore_index=True)
cond_map = dict(zip(wi_df['window_id'].astype(str), wi_df['condition'].astype(str)))
orig_map = dict(zip(wi_df['window_id'].astype(str), wi_df['forecast_origin'].astype(str)))
panel_map = dict(zip(wi_df['window_id'].astype(str), wi_df['panel'].astype(str)))

s44['condition'] = s44['window_id'].map(cond_map)
s44['panel'] = s44['window_id'].map(panel_map)
s44['h'] = ((pd.to_datetime(s44['timestamp'], utc=True) - pd.to_datetime(s44['window_id'].map(orig_map), utc=True)).dt.total_seconds() // 300).astype(int)
s44['split'] = np.where(s44['timestamp'].str[:7] == '2031-03', 'validation', 'private')

p_keys = ['window_id', 'h', 'link_id']
for k in p_keys:
    probs[k] = probs[k].astype(str)
    s44[k] = s44[k].astype(str)

m = s44.merge(probs[p_keys + ['p_s24']], on=p_keys, how='inner')

for split in ['validation', 'private']:
    sp_df = m[(m['split'] == split) & (m['condition'] == 'queue_ongoing')].copy()
    panels = sp_df['panel'].unique()
    
    print(f"\n{'='*20} {split.upper()} ISOLATED SPIKES [0, 1, 0] {'='*20}")
    spikes_found = []
    
    for p in panels:
        topo_path = DATA / "corridors" / p / "network" / "lwr_mainline_topology.csv"
        if not topo_path.exists(): continue
        topo = pd.read_csv(topo_path)
        ordered_links = topo['link_id'].astype(str).tolist()
        
        p_sub = sp_df[sp_df['panel'] == p]
        for (wid, h), grp in p_sub.groupby(['window_id', 'h']):
            q_map = dict(zip(grp['link_id'], grp['queue_pred']))
            p_map = dict(zip(grp['link_id'], grp['p_s24']))
            idx_map = dict(zip(grp['link_id'], grp.index))
            
            arr = np.array([q_map.get(l, 0) for l in ordered_links])
            spike_mask = (arr[:-2] == 0) & (arr[1:-1] == 1) & (arr[2:] == 0)
            
            if spike_mask.any():
                for idx in np.where(spike_mask)[0]:
                    mid_link = ordered_links[idx + 1]
                    spikes_found.append({
                        'window_id': wid,
                        'h': h,
                        'link_id': mid_link,
                        'panel': p,
                        'prob': p_map.get(mid_link, 0.0),
                    })
                    
    sp_df_spikes = pd.DataFrame(spikes_found)
    print(f"Total isolated spikes [0, 1, 0]: {len(sp_df_spikes)}")
    if not sp_df_spikes.empty:
        print("Prob distribution of spikes:")
        print(f"  min={sp_df_spikes['prob'].min():.3f}, mean={sp_df_spikes['prob'].mean():.3f}, median={sp_df_spikes['prob'].median():.3f}, max={sp_df_spikes['prob'].max():.3f}")
        for thresh in [0.30, 0.35, 0.40, 0.45]:
            count = (sp_df_spikes['prob'] < thresh).sum()
            print(f"  Spikes with prob < {thresh:.2f}: {count}")

