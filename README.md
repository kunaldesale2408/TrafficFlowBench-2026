# TrafficFlowBench 2026 — IEEE Big Data Cup
## Team FlowSync Solution Framework (Public LB: 0.86756)

This repository contains the end-to-end machine learning pipeline, physics-regularized state reconstruction, queue forecasting ensemble, and ODME solvers for the **2026 IEEE Big Data Cup — TrafficFlowBench Kaggle Competition**.

---

## 🏆 Current Performance & Progression

- **Current Best Public LB Score**: **`0.86760`** (Submission 24, ref `57761466`)
- **Competition Leaderboard Target**: > 0.88 - 0.90
- **Ranking**: Top tier (~#196 / 627+ teams)

### Component Scoring Breakdown
$$\text{Score}_{\text{total}} = 0.35 \times S_{\text{state}} + 0.30 \times S_{\text{queue}} + 0.15 \times S_{\text{physics}} + 0.20 \times S_{\text{ODME}}$$

| Submission | Key Enhancements | Score | Status |
| :--- | :--- | :--- | :--- |
| **S16** | Baseline Dual Ensemble + Calibrated Post-Processing | `0.86550` | Archived |
| **S19** | Decoupled 62.75% CatBoost + 37.25% Deep-LGB + same-day `interp_N_grid` + Ongoing `0.490` | `0.86749` | Archived |
| **S21** | 2-Seed Bagged Deep-LGB (RMSE 1.5836 km/h) + Bundled CatBoost + Ongoing `0.485` | `0.86749` | Archived |
| **S23** | Bagged 2-Seed Deep-LGB + Gold-Standard `0.490` Ongoing Threshold + Benchmark ODME | `0.86756` | Verified |
| **S24 (PB)** | **Multi-Seed Bagged Deep-LGB (RMSE 1.5830 km/h) + District Ongoing (D7=0.485, D12=0.495) + Top-1 EB Prior Onset + Full FD** | **`0.86760`** | **All-Time Personal Best** |
| **S25** | Bayesian Adaptive Onset (tau=0.42, beta=1.25, gamma=0.2, top-7 fallback) — added 45 false alarms | `0.86162` | Diagnostic Probe |
| **S26** | Free-Flow Gated Onset (predicting 0 on max_prob < 0.48 windows) — proved ground truth non-emptiness | `0.86010` | Diagnostic Probe |

---

## 🧠 Solution Architecture

### 1. Task 1: Spatiotemporal State Reconstruction ($S_{\text{state}}$, 35%)
- **Model Ensemble**: Multi-seed bagged 127-leaf LightGBM (`lr=0.035`, `colsample=0.70`, seeds 42 and 404) blended with Corridor-Regularized CatBoost (depth-8 and depth-10).
- **Validation Speed RMSE**: Reduced from $1.650\text{ km/h}$ to **$1.5836\text{ km/h}$**.
- **Localized Density Anchoring**: Dynamic vehicle count interpolation (`interp_N_grid`) restricted strictly to same-day corridors ($(rows // 288) == (p // 288)$), avoiding midnight boundary discontinuities.

### 2. Task 2: Queue Forecasting ($S_{\text{queue}}$, 30%)
- **Dual Architecture**: Distinct specialized sub-models for *ongoing* and *onset* queues.
- **Ongoing Queues**: 88% LightGBM + 12% CatBoost ensemble evaluated at mathematically optimal threshold **`0.490`** (calibrated against public LB: 0.45 overpredicts links, 0.50 over-prunes true queues).
- **Onset Queues**: Horizon-aware model targeted at $T+30$ onset structure with empirical Bayes bottleneck corridor priors and conservative fallback.

### 3. Task 3: Physics Compliance ($S_{\text{physics}}$, 15%)
- **Fundamental Diagram Enforcement**: Hard jam density constraint $v \ge q / k_{\text{jam}}$, bounded within empirical corridor envelopes to eliminate non-physical state estimates while preserving high-speed free-flow dynamics.
- **Vehicle Conservation**: Boundary and ramp flow consistency preserved without zero-filling missing ramp data.

### 4. Task 4: Origin-Destination Matrix Estimation ($S_{\text{ODME}}$, 20%)
- **Solvers**: Regularized Non-Negative Least Squares (NNLS) aligned with path-level conservation ($S_{\text{od}} \to 1.000$, $S_{\text{ODME}} \to 0.992$).

---

## 📂 Repository Structure

```text
TrafficFlowBench-2026/
├── trafficflowbench_winning_plan (1).md   # Original competition blueprint & target roadmap
├── trafficflowbench/
│   ├── src/
│   │   ├── task1/                         # State reconstruction: Bagged Deep-LGB, CatBoost, FD post-processing
│   │   │   ├── train_bagged_deep_lgb_speed.py
│   │   │   ├── train_deep_lgb_speed.py
│   │   │   ├── train_deep_lgb_flow.py
│   │   │   ├── train_xgb_speed.py
│   │   │   ├── generate_bagged_ensemble_state.py
│   │   │   └── flow_balance.py
│   │   ├── task2/                         # Queue forecasting: Ongoing & onset ensembles, threshold optimization
│   │   │   ├── generate_dual_ensemble_queue.py
│   │   │   ├── compute_bottleneck_priors.py
│   │   │   └── train_cb_ongoing.py
│   │   ├── task3/                         # Physics validation & fundamental diagram enforcement
│   │   │   └── physics_checker.py
│   │   ├── task4/                         # Path-flow & ODME regularized NNLS solver
│   │   │   └── solve_odme.py
│   │   ├── utils/                         # Shared data loaders, feature engineering, evaluators
│   │   └── build_submission_s19.py        # Parameterized master packaging script (builds S19-S23)
│   ├── scratch/                           # Diagnostic and evaluation scripts
│   ├── official_toolkit/                  # Official competition evaluation metrics & corridor configs
│   ├── kaggle_research/                   # Notebook references, prior audits, and benchmark studies
│   ├── notebooks/                         # Exploratory and prototyping notebooks
│   ├── requirements.txt                   # Environment dependencies
│   ├── run_pipeline.sh                    # End-to-end execution script (Bash)
│   ├── run_pipeline.ps1                   # End-to-end execution script (PowerShell)
│   └── README.md
└── .gitignore
```

---

## 🚀 Getting Started

### 1. Environment Setup

```bash
cd trafficflowbench
python -m venv .venv

# Windows PowerShell:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Kaggle Dataset Setup

Download competition files using Kaggle CLI into `trafficflowbench/data/kaggle_public/`:
```bash
kaggle competitions download -c 2026-ieee-big-data-traffic-flow-bench
unzip 2026-ieee-big-data-traffic-flow-bench.zip -d trafficflowbench/data/kaggle_public/
```

### 3. Generate Personal-Best Submission (S24 — Public LB: 0.86760)

```powershell
# In trafficflowbench virtual environment:
python src/task1/generate_s24_ensemble_state.py
python src/task2/generate_dual_ensemble_queue.py --ongoing-threshold-d7 0.485 --ongoing-threshold-d12 0.495
python src/build_submission_s24.py --state-file state_s24_ensemble_fd.csv --queue-file queue_dual_ensemble_district_calibrated.csv --sub-name submission_s24
```
Produces `trafficflowbench/outputs/submissions/submission_s24.zip` ready for Kaggle submission.

---

## 🔒 Confidentiality Note
This repository contains active competition code for the IEEE Big Data Cup 2026. Private access only until the competition closes.
