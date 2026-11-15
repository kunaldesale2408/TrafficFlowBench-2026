import json

p = "trafficflowbench/kaggle_research/anomalies/trafficflowbench-eleven-anomalies-of-the-data.ipynb"
nb = json.load(open(p, encoding='utf-8'))
print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    src = "".join(c.get('source', []))
    if c['cell_type'] == 'markdown':
        lines = [l.strip() for l in src.split('\n') if l.strip()]
        if any(w in src.lower() for w in ['anomaly', 'speed', 'flow', 'queue', 'odme', 'synthetic', 'generator']):
            print(f"[MD {i}]: {' | '.join(lines[:2])}")
