"""Generate Task 1 state predictions using Anubisyy's trained CatBoost models."""
from pathlib import Path
import os, sys, json, time
import numpy as np
import pandas as pd

DATA = Path("data/kaggle_public").resolve()
ASSETS = Path("kaggle_research/anubis_assets").resolve()
OUT = Path("outputs/submissions").resolve()
OUT.mkdir(parents=True, exist_ok=True)
THREADS = min(8, os.cpu_count() or 4)

os.environ["TFB_DATA_ROOT"] = str(DATA)
os.environ["TFB_OUTPUT_ROOT"] = str(OUT)
os.environ["TFB_THREADS"] = str(THREADS)
os.environ["TFB_PREDICT_DEVICE"] = "CPU"

sys.path.insert(0, str(ASSETS))
import state_model as base
from prepare_state import features
from state_inference import StatePredictor

def main():
    t0 = time.time()
    print("Loading CatBoost StatePredictor models...")
    predictor = StatePredictor(ASSETS)
    out_file = OUT / "state_catboost.csv"
    first = True
    total_rows = 0

    splits = ("validation", "private")
    panels = base.all_panels()
    print(f"Predicting for splits: {splits}, {len(panels)} panels...")

    for split in splits:
        for pi, name in enumerate(panels):
            t_start = time.time()
            panel = base.Panel(name, pi)
            frame = base.month_frame(name, '', split=split)
            times, v, q, e = base.matrices(frame, panel)
            target = pd.read_csv(base.RELEASE / 'task1' / name / split / 'sample_submission_state.csv',
                                 dtype={'station_id': str, 'link_id': str})
            rows = times.get_indexer(target.timestamp)
            cols = pd.Index(panel.links).get_indexer(target.link_id)
            assert (rows >= 0).all() and (cols >= 0).all()
            regime = target.mask_regime.map({'R1': 0, 'R2': 1, 'R3': 2}).to_numpy()
            
            X, bv, bq = features(panel, pi, times, v, q, e, rows, cols, regime)
            assert X.shape[1] == 94
            
            speed_pred = np.clip(bv + predictor.residual(X, 'speed'), 1.0, 180.0)
            flow_pred = np.clip((bq + predictor.residual(X, 'flow')) * panel.lanes[cols], 0.0, 150000.0)
            
            target['speed_kmh'] = speed_pred
            target['flow_vph'] = flow_pred
            
            cols_to_write = ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime', 'speed_kmh', 'flow_vph']
            target[cols_to_write].to_csv(out_file, index=False, mode='w' if first else 'a', header=first)
            first = False
            total_rows += len(target)
            print(f"  [{name} {split}] wrote {len(target):,} rows in {time.time()-t_start:.1f}s (total: {total_rows:,})", flush=True)

    print(f"\nAll done in {time.time()-t0:.1f}s! Wrote {total_rows:,} rows to {out_file}")

if __name__ == "__main__":
    main()
