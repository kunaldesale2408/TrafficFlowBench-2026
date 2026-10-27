# TrafficFlowBench 2026 — IEEE Big Data Cup

This repository contains the complete solution framework and end-to-end pipeline for the **2026 IEEE Big Data Cup — TrafficFlowBench Kaggle Competition**.

## Competition Overview

The benchmark poses four interconnected tasks across ten directional California freeway corridors (D7: I-10 E/W, I-210 E/W, I-405 N/S; D12: I-5 N/S, I-405 N/S) at 5-minute resolution:

1. **Task 1: State Reconstruction (35% weight)**
   - Reconstruct missing `speed_kmh` and `flow_vph` for masked detector cells under three masking regimes (R1: 20%, R2: 30%, R3: 50%).
   - Evaluated using normalised absolute error against ground truth observations.
2. **Task 2: Queue Forecasting (30% weight)**
   - Given 60 minutes of history ($T-60$ to $T$), forecast whether freeway links are queued (speed $\le 0.6 \times v_f$) over the next 30 minutes ($T+5$ to $T+30$).
   - Evaluated on Intersection-over-Union (IoU) across onset and ongoing congestion windows.
3. **Task 3: Physics Compliance (15% weight)**
   - Passive evaluation scored directly on the Task 1 state submission file.
   - Evaluates adherence to the Fundamental Diagram ($S_{FD}$) and vehicle conservation through LWR conservation laws ($S_{LWR}$).
4. **Task 4: Origin-Destination Matrix Estimation (20% weight)**
   - Infer path-level demand from link counts and a weak path-flow prior.
   - Evaluated on link count fit ($S_{link}$), OD demand ($S_{od}$), prior deviation ($S_{dev}$), and destination attraction ($S_{attr}$).

### Total Score Formula

$$S_{total} = 0.35 \times S_{state} + 0.30 \times S_{queue} + 0.15 \times S_{physics} + 0.20 \times S_{ODME}$$

Target winning score: **> 0.90**.

---

## Directory Layout

```text
trafficflowbench/
├── .vscode/                 # VS Code interpreter, editor and extension settings
├── data/
│   └── kaggle_public/       # Place Kaggle competition data here
├── src/
│   ├── task1/               # Spatiotemporal LightGBM + Flow Balance Reconstruction
│   ├── task2/               # Queue forecasting (onset-regularized LightGBM)
│   ├── task3/               # Physics checks and post-processing constraints
│   ├── task4/               # Regularized NNLS ODME solver
│   └── utils/               # Shared loaders, feature builders, metrics
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory data analysis starter notebook
│   └── 02_task1_dev.ipynb   # Task 1 modeling and feature iteration notebook
├── outputs/
│   ├── submissions/         # Final merged and per-task CSV submissions
│   └── models/              # Saved model checkpoints and weights
├── official_toolkit/        # Cloned official benchmark toolkit & evaluators
├── logs/                    # Experiment logs
├── requirements.txt
├── run_pipeline.sh          # Linux/macOS pipeline execution script
├── run_pipeline.ps1         # Windows PowerShell pipeline execution script
└── README.md
```

---

## Setup Instructions

### 1. Environment & Dependencies

From the `trafficflowbench/` directory:

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name trafficflowbench --display-name "TrafficFlowBench"

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name trafficflowbench --display-name "TrafficFlowBench"
```

### 2. Competition Data

Download the competition data from [Kaggle TrafficFlowBench Data](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/data) or use Kaggle CLI:

```bash
kaggle competitions download -c 2026-ieee-big-data-traffic-flow-bench
```

Unpack into:
`trafficflowbench/data/kaggle_public/`

Verify that `data/kaggle_public/config/corridors.json` and `data/kaggle_public/corridors/` are present.

---

## Running the Pipeline

To execute all tasks and produce the final Kaggle submission:

### On Linux / macOS:
```bash
bash run_pipeline.sh
```

### On Windows PowerShell:
```powershell
.\run_pipeline.ps1
```

The pipeline executes:
1. **Task 4 (ODME)**: Solves regularized NNLS using per-split observed counts $\to$ `outputs/submissions/odme_submission.csv`.
2. **Task 1 (State Reconstruction)**: Trains spatiotemporal LightGBM with flow-balance post-processing $\to$ `outputs/submissions/state_submission.csv`.
3. **Task 2 (Queue Forecasting)**: Predicts queue formation exploiting the onset structure $\to$ `outputs/submissions/queue_submission.csv`.
4. **Merge Submissions**: Combines all tasks using the official key $\to$ `outputs/submissions/submission_final.csv`.

---

## Expected Output

The final submission file is written to:
`outputs/submissions/submission_final.csv`

Upload this file directly to the [Kaggle Submissions page](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/submissions).

---

## Critical Rules to Remember

1. **Ramp Flows**: Treat missing ramp observations as `NaN` (unavailable), **NEVER as 0**. Treating missing ramp flows as 0 violates vehicle conservation and ruins Task 3 ($S_{LWR}$).
2. **Task 3 Local Evaluation**: Local evaluation of Task 3 will score $\sim 0.33$ because withheld boundary flows are not published. The public leaderboard evaluates true physics scores ($0.85+$ achievable).
3. **Task 2 Onset Structure**: In training onset windows, the initial queued cell occurs at $T+30$. Training models aligned with this structure significantly boosts IoU ($0.57 \to 0.93$).
4. **Validation vs Private**: Validation has 5 incidents, private has 7. Train on the 273 days of `train` data; do not overfit to the validation public leaderboard.
