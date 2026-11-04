"""Extract exact ground-truth queue states for all 80 official train windows.
This allows 100% accurate offline validation and IoU calculation for Task 2.
"""
from pathlib import Path
import pandas as pd
import numpy as np

def extract_train_ground_truth(root: Path = Path("trafficflowbench/data/kaggle_public")):
    panels = [d.name for d in (root / "corridors").iterdir() if d.is_dir() and d.name not in {"D12_I405_N", "D12_I405_S"}]
    print(f"Extracting GT for {len(panels)} panels: {panels}")

    gt_records = []
    for p in panels:
        wi_path = root / "task2" / p / "train" / "window_index.csv"
        if not wi_path.exists():
            continue
        wi = pd.read_csv(wi_path)
        
        fd_path = root / "corridors" / p / "network" / "fd_parameters.csv"
        fd = pd.read_csv(fd_path).drop_duplicates("link_id")
        v_cut_col = "v_cut" if "v_cut" in fd.columns else None
        v_cut_map = dict(zip(fd.link_id.astype(str), fd[v_cut_col] if v_cut_col else 0.6 * fd.free_speed_kmh))
        
        for row in wi.itertuples():
            wid = str(row.window_id)
            d = str(row.date)
            t_orig = pd.Timestamp(row.forecast_origin)
            ym = d[:7]
            clean_date = d.replace("-", "_")
            fname = f"synthetic_mainline_{clean_date}.parquet"
            fpath = root / "corridors" / p / "train" / "mainline_states" / f"year_month={ym}" / fname
            if not fpath.exists():
                print(f"Warning: {fpath} not found!")
                continue
            df = pd.read_parquet(fpath)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            target_times = [t_orig + pd.Timedelta(minutes=5*h) for h in range(1, 7)]
            sub_df = df[df["timestamp"].isin(target_times)].copy()
            sub_df["window_id"] = wid
            sub_df["panel"] = p
            sub_df["condition"] = row.condition
            sub_df["link_id"] = sub_df["link_id"].astype(str)
            sub_df["v_cut"] = sub_df["link_id"].map(v_cut_map)
            sub_df["queue_true"] = ((sub_df["speed_kmh"] <= sub_df["v_cut"]) & sub_df["is_score_eligible"].astype(bool)).astype(int)
            gt_records.append(sub_df[["window_id", "panel", "condition", "timestamp", "link_id", "speed_kmh", "flow_vph", "is_score_eligible", "queue_true"]])

    full_gt = pd.concat(gt_records, ignore_index=True)
    out_dir = root.parent.parent / "outputs" / "ground_truth"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "task2_train_ground_truth.parquet"
    full_gt.to_parquet(out_file, index=False)
    print(f"Extracted {len(full_gt):,} records for {full_gt['window_id'].nunique()} windows.")
    print(f"Saved to {out_file}")
    print("\nQueue distribution:")
    print(full_gt.groupby("condition")["queue_true"].mean())
    return full_gt

if __name__ == "__main__":
    extract_train_ground_truth()
