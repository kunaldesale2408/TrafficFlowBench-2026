import json

path = "trafficflowbench/kaggle_research/sachith/trafficflowbench-when-does-temporal-repair-fail.ipynb"
nb = json.load(open(path, "r", encoding="utf-8"))
print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'markdown':
        src = "".join(c.get('source', '')).strip().replace('\n', ' ')
        print(f"[{i}] MD: {src[:120]}")
    else:
        src = "".join(c.get('source', '')).strip().replace('\n', ' ')
        print(f"[{i}] CODE: {src[:80]}")
