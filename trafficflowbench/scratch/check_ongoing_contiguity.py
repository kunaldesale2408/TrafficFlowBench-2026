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

p_keys = ['window_id', 'h', 'link_id']
for k in p_keys:
    probs[k] = probs[k].astype(str)
    s44[k] = s44[k].astype(str)

m = s44.merge(probs[p_keys + ['split', 'p_s24']], on=p_keys, how='inner')

# Filter to validation ongoing
val_ong = m[(m['split'] == 'validation') & (m['condition'] == 'queue_ongoing')].copy()
panels = val_ong['panel'].unique()

print("Analyzing spatial contiguity of candidate queue extensions...")

for tau in [0.42, 0.40, 0.38, 0.35]:
    cand_mask = (val_ong['queue_pred'] == 0) & (val_ong['p_s24'] >= tau)
    total_cands = cand_mask.sum()
    
    adjacent_count = 0
    isolated_count = 0
    
    for p in panels:
        topo_path = DATA / "corridors" / p / "network" / "lwr_mainline_topology.csv"
        if not topo_path.exists(): continue
        topo = pd.read_csv(topo_path)
        ordered_links = topo['link_id'].astype(str).tolist()
        link_pos = {l: i for i, l in enumerate(ordered_links)}
        
        p_df = val_ong[val_ong['panel'] == p]
        for (wid, h), grp in p_df.groupby(['window_id', 'h']):
            q_set = set(grp[grp['queue_pred'] == 1]['link_id'])
            cands = grp[(grp['queue_pred'] == 0) & (grp['p_s24'] >= tau)]['link_id'].tolist()
            
            for cl in cands:
                pos = link_pos.get(cl)
                if pos is None:
                    isolated_count += 1
                    continue
                # Check 1-hop upstream or downstream
                up = ordered_links[pos - 1] if pos > 0 else None
                dn = ordered_links[pos + 1] if pos < len(ordered_links) - 1 else None
                
                if (up in q_set) or (dn in q_set):
                    adjacent_count += 1
                else:
                    isolated_count += 1
                    
    print(f"tau={tau:.2f} | Total: {total_cands} | Adjacent to Existing Queue: {adjacent_count} ({adjacent_count/max(total_cands,1):.1%}) | Isolated: {isolated_count}")

