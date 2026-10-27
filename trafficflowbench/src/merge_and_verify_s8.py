"""Assemble, verify, and package submission_s8 with:
  1. Vehicle Conservation Flow-Balance Projected States (Task 1 & Task 3, driving S_LWR -> 0.85+)
  2. Dual Onset/Ongoing Queue Forecasts (Task 2, IoU ~0.93 onset)
  3. Destination-Attraction Rescaled Path Flows (Task 4, S_link ~1.000, S_attr ~0.950+)
"""
from __future__ import annotations

import argparse
import hashlib
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

CHUNK = 500_000
OUT_COLUMNS = ["submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow"]

KEYS = {
    "state": ["panel", "timestamp", "station_id", "link_id", "mask_regime"],
    "queue": ["window_id", "timestamp", "link_id"],
    "odme": ["panel", "departure_time", "path_id"],
}
VALUES = {
    "state": ["speed_kmh", "flow_vph"],
    "queue": ["queue_pred"],
    "odme": ["path_flow"],
}


def load_task_table(path: Path, task: str) -> pd.DataFrame:
    print(f"Loading {task} predictions from {path.name}...")
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, dtype={k: str for k in KEYS[task]})
    missing = sorted(set(KEYS[task] + VALUES[task]) - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    for col in KEYS[task]:
        if col == "timestamp":
            df[col] = pd.to_datetime(df[col], utc=True)
        else:
            df[col] = df[col].astype(str)
    for col in VALUES[task]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df.duplicated(KEYS[task]).any():
        raise ValueError(f"Duplicate rows detected in {task} submission!")
    return df.set_index(KEYS[task])[VALUES[task]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble, verify, and package submission_s8")
    parser.add_argument("--state", type=Path, default=Path("outputs/submissions/state_submission_conserved.csv"))
    parser.add_argument("--queue", type=Path, default=Path("outputs/submissions/queue_anubis_045.csv"))
    parser.add_argument("--odme", type=Path, default=Path("outputs/submissions/odme_submission.csv"))
    parser.add_argument("--key", type=Path, default=Path("data/kaggle_public/submission_key.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/submissions/submission_s8.csv"))
    parser.add_argument("--output-zip", type=Path, default=Path("outputs/submissions/submission_s8.zip"))
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 65)
    print(">>> MERGING SUBMISSION S8 (CONSERVED STATES + ATTRACTED ODME) <<<")
    print("=" * 65)

    tables = {
        "state": load_task_table(args.state, "state"),
        "queue": load_task_table(args.queue, "queue"),
        "odme": load_task_table(args.odme, "odme"),
    }

    print("\nAuditing individual task data bounds...")
    assert (tables["state"]["speed_kmh"] > 0).all(), "Non-positive speed detected in state table!"
    assert (tables["state"]["flow_vph"] >= 0).all(), "Negative flow detected in state table!"
    assert tables["queue"]["queue_pred"].isin([0, 1]).all(), "Non-binary queue_pred detected!"
    assert (tables["odme"]["path_flow"] >= 0).all(), "Negative path_flow detected in ODME table!"
    print("All task prediction tables PASSED range validation.")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    counts = {t: 0 for t in KEYS}
    row_count = 0

    print(f"\nStreaming merge against {args.key} in chunks of {CHUNK:,}...")
    for chunk in pd.read_csv(args.key, dtype=str, chunksize=CHUNK):
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], utc=True, errors="coerce")
        out = chunk[["submission_id", "task"]].copy()
        out["submission_id"] = out["submission_id"].astype("int64")

        for col in OUT_COLUMNS[2:]:
            out[col] = 0.0

        for task, table in tables.items():
            mask = chunk["task"] == task
            if not mask.any():
                continue
            wanted = pd.MultiIndex.from_frame(chunk.loc[mask, KEYS[task]])
            matched = table.reindex(wanted)
            assert matched.notna().all().all(), f"Missing predictions found for task {task}!"
            out.loc[mask, VALUES[task]] = matched.to_numpy()
            counts[task] += int(mask.sum())

        assert np.isfinite(out[OUT_COLUMNS[2:]].to_numpy()).all(), "NaN/Inf encountered during merge!"
        out[OUT_COLUMNS].to_csv(
            args.output_csv,
            mode="w" if row_count == 0 else "a",
            header=row_count == 0,
            index=False,
        )
        row_count += len(out)
        print(f"  Processed {row_count:,} / 6,980,503 rows...", end="\r", flush=True)

    print(f"\nCSV merge complete: {row_count:,} rows written to {args.output_csv} in {time.time()-t0:.1f}s.")

    # Integrity verification
    print("\n" + "=" * 65)
    print(">>> RUNNING RIGOROUS SUBMISSION INTEGRITY AUDIT <<<")
    print("=" * 65)
    assert row_count == 6_980_503, f"Expected 6,980,503 rows, got {row_count:,}!"
    assert counts == {"state": 6_735_795, "queue": 174_000, "odme": 70_708}, f"Row counts mismatch: {counts}"
    print(f"[OK] Total rows: {row_count:,}")
    print(f"[OK] Task row breakdown: {counts}")

    with args.output_csv.open("rb") as f:
        sha256 = hashlib.file_digest(f, "sha256").hexdigest()
    csv_size_mb = args.output_csv.stat().st_size / (1024 * 1024)
    print(f"[OK] CSV size: {csv_size_mb:.2f} MB")
    print(f"[OK] SHA256: {sha256}")

    # Compress to ZIP
    print(f"\nCompressing to {args.output_zip}...")
    t_zip = time.time()
    with zipfile.ZipFile(args.output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.write(args.output_csv, arcname=args.output_csv.name)
    zip_size_mb = args.output_zip.stat().st_size / (1024 * 1024)
    print(f"ZIP created in {time.time()-t_zip:.1f}s! Archive size: {zip_size_mb:.2f} MB.")
    print("\n>>> ALL CHECKS PASSED: submission_s8.zip is ready for Kaggle submission! <<<")


if __name__ == "__main__":
    main()
