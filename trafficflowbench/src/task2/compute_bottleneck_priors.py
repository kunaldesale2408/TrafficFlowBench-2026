"""Compute Empirical Bayes bottleneck breakdown priors for all corridors.

Calculates:
  p_EB(link, slot) = (hits + M * p_bar_corridor(slot)) / (N + M)
where slot = weekday * 288 + tod (2,016 weekly 5-min intervals).
Used to guide and calibrate Task 2 onset bottleneck localization.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("trafficflowbench").resolve()
DATA = ROOT / "data" / "kaggle_public"
OUT_DIR = ROOT / "outputs" / "priors"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SLOTS = 7 * 288
SHRINK_M = 20.0


def slot_of(ts: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(ts, utc=True)
    return (ts.dt.weekday * 288 + ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(dtype=np.int32)


def load_vcut_map(panel_dir: Path) -> dict[str, float]:
    fd_file = panel_dir / "network" / "fd_parameters.csv"
    if not fd_file.exists():
        return {}
    df = pd.read_csv(fd_file, dtype={"link_id": str})
    vcut = {}
    for _, r in df.iterrows():
        # Critical speed v_c or v_cut
        vc = float(r.get("critical_speed_kmh", r.get("v_cut", 60.0)))
        vcut[str(r["link_id"])] = vc
    return vcut


def build_corridor_bottleneck_prior(panel: str) -> dict:
    t0 = time.time()
    pdir = DATA / "corridors" / panel
    vcut_map = load_vcut_map(pdir)
    train_files = sorted((pdir / "train" / "mainline_states").glob("**/*.parquet"))
    
    hits: dict[tuple[str, int], int] = {}
    tot_link_slot: dict[tuple[str, int], int] = {}
    hits_slot = np.zeros(N_SLOTS, dtype=np.float64)
    tot_slot = np.zeros(N_SLOTS, dtype=np.float64)
    
    for f in train_files:
        try:
            df = pd.read_parquet(f, columns=["timestamp", "link_id", "speed_kmh"])
        except Exception:
            continue
        df["link_id"] = df["link_id"].astype(str)
        sp = pd.to_numeric(df["speed_kmh"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(sp)
        if not ok.any():
            continue
        
        slot = slot_of(df["timestamp"])
        cuts = np.array([vcut_map.get(str(lid), 60.0) for lid in df["link_id"]], dtype=float)
        q = (ok & (sp <= cuts)).astype(np.int32)
        
        for lid, s, qq, oo in zip(df["link_id"].to_numpy(), slot, q, ok.astype(np.int32)):
            k = (str(lid), int(s))
            hits[k] = hits.get(k, 0) + int(qq)
            tot_link_slot[k] = tot_link_slot.get(k, 0) + int(oo)
        
        np.add.at(hits_slot, slot[ok], q[ok])
        np.add.at(tot_slot, slot[ok], 1)
        
    p_bar = hits_slot / np.maximum(tot_slot, 1.0)
    
    # Per-link EB breakdown probabilities
    by_link: dict[str, dict[int, float]] = {}
    overall_link_hits: dict[str, int] = {}
    overall_link_tot: dict[str, int] = {}
    
    for (lid, s), h in hits.items():
        n = tot_link_slot.get((lid, s), 0)
        if n == 0:
            continue
        p_eb = (h + SHRINK_M * float(p_bar[s])) / (n + SHRINK_M)
        if lid not in by_link:
            by_link[lid] = {}
        by_link[lid][int(s)] = round(float(p_eb), 5)
        overall_link_hits[lid] = overall_link_hits.get(lid, 0) + h
        overall_link_tot[lid] = overall_link_tot.get(lid, 0) + n
    
    link_overall_freq = {
        lid: round(float(overall_link_hits[lid]) / max(overall_link_tot[lid], 1), 5)
        for lid in overall_link_hits
    }
    
    result = {
        "panel": panel,
        "p_bar_corridor": [round(float(x), 5) for x in p_bar],
        "links": by_link,
        "link_overall_freq": link_overall_freq,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    
    out_file = OUT_DIR / f"{panel}_bottleneck_prior.json"
    out_file.write_text(json.dumps(result))
    print(f"[{panel}] Bottleneck prior computed in {time.time()-t0:.1f}s -> {out_file.name}")
    return result


def main():
    print("=" * 65)
    print(">>> COMPUTING EMPIRICAL BOTTLENECK BREAKDOWN PRIORS <<<")
    print("=" * 65)
    
    panels = [p.name for p in (DATA / "corridors").iterdir() if p.is_dir()]
    # Task 2 is evaluated on 8 corridors (excluding D12_I405_N and D12_I405_S)
    task2_panels = [p for p in panels if p not in {"D12_I405_N", "D12_I405_S"}]
    print(f"Target corridors ({len(task2_panels)}): {task2_panels}")
    
    for panel in task2_panels:
        build_corridor_bottleneck_prior(panel)
        
    print("\nAll bottleneck priors successfully computed and cached in outputs/priors/.")


if __name__ == "__main__":
    main()
