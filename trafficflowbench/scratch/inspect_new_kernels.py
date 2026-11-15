import json
import glob
from pathlib import Path

def summarize_nb(path):
    print("=" * 70)
    print(f"FILE: {path}")
    print("=" * 70)
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"Total cells: {len(nb['cells'])}")
    for i, c in enumerate(nb['cells']):
        src = "".join(c.get("source", "")).strip()
        lines = [line.strip() for line in src.split("\n") if line.strip()]
        header = lines[0] if lines else ""
        if c['cell_type'] == 'markdown':
            print(f"[{i}] MD: {header[:80]}")
        else:
            print(f"[{i}] CODE ({len(lines)} lines): {header[:80]}")

for p in ["trafficflowbench/kaggle_research/autokagon/public-fast-gpu-inference.ipynb",
          "trafficflowbench/kaggle_research/plarmuseau_jam/trafficflowbench-the-physics-of-a-synthetic-jam.ipynb"]:
    if Path(p).exists():
        summarize_nb(p)

for p in ['trafficflowbench/kaggle_research/cacinie/tfb-v5orig.ipynb',
          'trafficflowbench/kaggle_research/prvsiyan/traffic-flow-trying-some-stuff.ipynb']:
    summarize_nb(p)
