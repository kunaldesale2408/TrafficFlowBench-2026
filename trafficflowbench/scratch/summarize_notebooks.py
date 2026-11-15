import json
from pathlib import Path

notebooks = [
    Path("trafficflowbench/kaggle_research/sujan_090/trafficflowbench-top-rank-pipeline-0-90-target.ipynb"),
    Path("trafficflowbench/kaggle_research/cacinie/tfb-v5orig.ipynb"),
    Path("trafficflowbench/kaggle_research/tyz/new-benchmark-task-875e8.ipynb"),
    Path("trafficflowbench/kaggle_research/tyz/new-benchmark-task-9b6ff.ipynb"),
]

for nb_path in notebooks:
    if not nb_path.exists():
        continue
    print(f"\n{'='*30} {nb_path.name} {'='*30}")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb.get("cells", [])
    print(f"Total cells: {len(cells)}")
    for i, cell in enumerate(cells):
        cell_type = cell.get("cell_type", "")
        source = "".join(cell.get("source", []))
        if cell_type == "markdown":
            lines = [line.strip() for line in source.split("\n") if line.strip()]
            if lines:
                print(f"  [MD {i}] " + " | ".join(lines[:3]))
        elif cell_type == "code":
            lines = [l for l in source.split("\n") if l.strip().startswith("#") or "score" in l or "read" in l or "import" in l or "def " in l]
            if lines:
                print(f"  [Code {i}] " + "; ".join([l.strip() for l in lines[:2]]))
