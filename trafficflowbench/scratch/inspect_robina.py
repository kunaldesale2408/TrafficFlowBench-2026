import json

path = "trafficflowbench/kaggle_research/robina_physics/trafficflowbench-physics-guided-state-odme.ipynb"
nb = json.load(open(path, "r", encoding="utf-8"))
print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    src = "".join(c.get('source', '')).strip().replace('\n', ' ')
    if c['cell_type'] == 'markdown':
        print(f"[{i}] MD: {src[:120]}")
    else:
        print(f"[{i}] CODE: {src[:80]}")
