import json

path = "trafficflowbench/kaggle_research/honghanhhh/trafficflowbench-lb-0-87141.ipynb"
nb = json.load(open(path, "r", encoding="utf-8"))
print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    src = "".join(c.get('source', '')).strip().replace('\n', ' ')
    print(f"[{i}] {c['cell_type'].upper()}: {src[:150]}")
