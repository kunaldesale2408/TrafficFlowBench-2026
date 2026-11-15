import json

with open("trafficflowbench/kaggle_research/anomalies/trafficflowbench-eleven-anomalies-of-the-data.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print("CELLS COUNT:", len(nb["cells"]))
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", "")).strip()
    lines = src.split("\n")
    print(f"--- CELL {i} ({c['cell_type']}, {len(lines)} lines) ---")
    for line in lines[:8]:
        print(" ", line)
