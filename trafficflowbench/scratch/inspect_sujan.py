import json
import glob
from pathlib import Path

def inspect_notebook(path):
    print("=" * 60)
    print(f"FILE: {path}")
    print("=" * 60)
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", "")).strip()
        if not src:
            continue
        lines = src.split("\n")
        print(f"--- CELL {i} ({cell['cell_type']}, {len(lines)} lines) ---")
        for line in lines[:8]:
            print(f"  {line}")
        if len(lines) > 8:
            print(f"  ... ({len(lines)-8} more lines)")

inspect_notebook("trafficflowbench/kaggle_research/semwal_087/trafficflowbench-trees-u-net-cnn-lb-0-87.ipynb")
inspect_notebook("trafficflowbench/kaggle_research/robina_physics/trafficflowbench-physics-guided-state-odme.ipynb")


