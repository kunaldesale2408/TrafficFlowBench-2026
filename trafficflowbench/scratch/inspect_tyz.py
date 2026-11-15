import json
import glob

for path in sorted(glob.glob("trafficflowbench/kaggle_research/tyz/*.ipynb")):
    nb = json.load(open(path, "r", encoding="utf-8"))
    print(f"=== {path} (total cells: {len(nb['cells'])}) ===")
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", "")).strip()
        first_line = src.split("\n")[0] if src else ""
        print(f"  Cell {i} ({c['cell_type']}): {first_line[:100]}")
