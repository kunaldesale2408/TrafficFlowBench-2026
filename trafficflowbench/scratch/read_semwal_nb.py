import json

nb = json.load(open('trafficflowbench/kaggle_research/semwal_087/trafficflowbench-trees-u-net-cnn-lb-0-87.ipynb', encoding='utf-8'))
for i, cell in enumerate(nb['cells']):
    source = "".join(cell.get('source', []))
    if cell['cell_type'] == 'markdown':
        print(f"[MD {i}]: {source[:200]}")
    elif cell['cell_type'] == 'code':
        first_lines = [l for l in source.split('\n') if l.strip()][:3]
        print(f"[Code {i}]: {'; '.join(first_lines)}")
