import json

def inspect(path):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print("CELL COUNT:", len(nb["cells"]))
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", "")).strip()
        lines = src.split("\n")
        print(f"--- CELL {i} ({c['cell_type']}, {len(lines)} lines) ---")
        for line in lines[:10]:
            print("  ", line)

inspect("trafficflowbench/kaggle_research/autokagon/public-fast-gpu-inference.ipynb")

