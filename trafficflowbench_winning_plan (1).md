# TrafficFlowBench 2026 — Complete Winning Plan
## 2026 IEEE Big Data Cup · Kaggle · Deadline ~Nov 6, 2026

**Target score: >0.90 total** (public LB leader is 0.845 as of Oct 2026)  
**Scoring formula:**
```
S_total = 0.35 × S_state + 0.30 × S_queue + 0.15 × S_physics + 0.20 × S_ODME
```

---

## FIRST PROMPT — Give This to Gemini CLI Before Anything Else

Paste this prompt into Gemini CLI verbatim. It creates the entire project as a
structured VS Code workspace with a virtual environment, folder layout, config
files, and all dependencies installed and ready to run.

```
Create a complete VS Code project for the 2026 IEEE Big Data Cup — TrafficFlowBench
Kaggle competition. Do the following in order:

1. CREATE PROJECT FOLDER STRUCTURE
   Create a root folder called trafficflowbench/ with this layout:
   trafficflowbench/
   ├── .vscode/
   │   ├── settings.json        # Python interpreter, linting, formatting
   │   └── extensions.json      # Recommended extensions list
   ├── data/
   │   └── kaggle_public/       # Empty placeholder — user pastes Kaggle data here
   ├── src/
   │   ├── task1/               # State reconstruction scripts
   │   ├── task2/               # Queue forecasting scripts
   │   ├── task3/               # Physics check (passive — reads Task 1 output)
   │   ├── task4/               # ODME scripts
   │   └── utils/               # Shared helpers (data loading, feature utils)
   ├── notebooks/
   │   ├── 01_eda.ipynb         # Exploratory data analysis starter notebook
   │   └── 02_task1_dev.ipynb   # Task 1 development notebook
   ├── outputs/
   │   ├── submissions/         # Final CSV files to upload to Kaggle
   │   └── models/              # Saved LightGBM model files (.pkl / .txt)
   ├── logs/                    # Experiment logs
   ├── .gitignore
   ├── requirements.txt
   ├── README.md
   └── run_pipeline.sh          # One-click script to rebuild submission end-to-end

2. CREATE VIRTUAL ENVIRONMENT AND INSTALL DEPENDENCIES
   Run these commands:
     python -m venv .venv
     source .venv/bin/activate   (or .venv\Scripts\activate on Windows)
     pip install --upgrade pip
     pip install lightgbm catboost scikit-learn pandas pyarrow polars \
                 scipy numba tqdm jupyter ipykernel matplotlib seaborn \
                 kaggle pyyaml joblib

   Then register the kernel:
     python -m ipykernel install --user --name trafficflowbench --display-name "TrafficFlowBench"

3. CLONE THE OFFICIAL TOOLKIT inside the project root:
     git clone https://github.com/jacky850/trafficflowbench-public.git official_toolkit/
   This gives access to the official baselines, scoring scripts, and data schemas.
   Do NOT modify files inside official_toolkit/. Copy scripts you need into src/.

4. CREATE .vscode/settings.json with this content:
   {
     "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
     "python.terminal.activateEnvironment": true,
     "editor.formatOnSave": true,
     "python.formatting.provider": "black",
     "jupyter.kernelspec.display.name": "TrafficFlowBench",
     "files.exclude": { "**/__pycache__": true, "**/*.pyc": true }
   }

5. CREATE .vscode/extensions.json recommending:
   Python, Pylance, Jupyter, GitLens, Rainbow CSV, Excel Viewer

6. CREATE requirements.txt listing every package installed in step 2,
   pinned to the installed versions (use pip freeze > requirements.txt).

7. CREATE run_pipeline.sh that runs tasks in order:
   #!/bin/bash
   set -e
   source .venv/bin/activate
   REL=./data/kaggle_public
   echo "=== Task 4: ODME ==="
   python src/task4/solve_odme.py --release-root $REL --split validation
   echo "=== Task 1: State Reconstruction ==="
   python src/task1/train_and_predict.py --release-root $REL --split validation
   echo "=== Task 2: Queue Forecasting ==="
   python src/task2/train_and_predict.py --release-root $REL --split validation
   echo "=== Merge Submissions ==="
   python official_toolkit/src/merge_submissions.py \
     --state  outputs/submissions/state_submission.csv \
     --queue  outputs/submissions/queue_submission.csv \
     --odme   outputs/submissions/odme_submission.csv \
     --output outputs/submissions/submission_final.csv
   echo "Done. Upload outputs/submissions/submission_final.csv to Kaggle."

8. CREATE src/utils/data_loader.py with a load_corridor() function that reads
   a single corridor's mainline_states_masked parquet/CSV, ramp flows, and config
   from kaggle_public/ and returns a clean pandas DataFrame with null ramp flows
   as NaN (never as 0).

9. CREATE a README.md that explains:
   - What this competition is (4 tasks, scoring formula)
   - How to set up (step-by-step from this prompt)
   - Where to paste the Kaggle data
   - How to run the full pipeline (run_pipeline.sh)
   - Expected output (submission_final.csv → upload to Kaggle)

10. VERIFY SETUP by running:
    python -c "import lightgbm, catboost, polars, scipy; print('All imports OK')"
    python official_toolkit/src/task1/build_task1_baseline_submission.py \
      --release-root ./data/kaggle_public --split train --panel D12_I5_N \
      --output outputs/submissions/baseline_state.csv
    If the data is not yet in data/kaggle_public/, print a clear message:
    "Paste your Kaggle competition data into data/kaggle_public/ and re-run."

When finished, open the trafficflowbench/ folder in VS Code and confirm the
Python interpreter in the bottom-left status bar shows '.venv'.
```

After Gemini completes this, your VS Code workspace is fully operational.
Paste the Kaggle data into `data/kaggle_public/` and move on to Task 4 (Section 5 below).

---

## 0. Critical Context — Read This First

These are the non-obvious facts that determine whether you win or lose:

1. **Task 3 (Physics) scores 0 locally** — the evaluator needs withheld boundary
   flows you don't have. You'll see 0.33 for everyone locally. The leaderboard
   gap between a perfect answer and the baseline is 0.9598 vs 0.3242. Focus on
   Task 1 quality; Task 3 comes free.

2. **The queue onset trick (now public knowledge):** In 39/40 training windows,
   the first queued cell appears exactly at T+30 due to how organizers chose
   window origins. Train Task 2 models on windows built the same way → onset IoU
   jumps from 0.57 → 0.93. This is confirmed-legal by the organizers. Use it.

3. **Validation ≠ Private split.** Validation has 5 incidents, private has 7, at
   different times and places. Do NOT overfit to public LB. Fit on `train`
   (273 days), use public LB as a transfer check only.

4. **Treat null ramps as unavailable, never as zero.** Treating missing ramp flow
   as 0 will destroy Task 3 (conservation).

5. **Flow is scored per lane.** A corridor with 6 lanes isn't penalized vs 3 lanes.
   Work per-lane, not per-link total.

6. **~24% of cells are never scored** (pct_observed < 75 or null). Avoid wasting
   compute on them.

---

## 1. Environment Setup

```bash
git clone https://github.com/jacky850/trafficflowbench-public.git
cd trafficflowbench-public
pip install -r requirements.txt
pip install lightgbm catboost scikit-learn pandas pyarrow polars scipy numba tqdm

# Download competition data from Kaggle → unpack to:
# ./data/kaggle_public/
REL=./data/kaggle_public
```

**Hardware:** Minimum 32 GB RAM, 8 cores. Task 2 trains on 56M rows — use
out-of-core or Polars/PyArrow for speed.

---

## 2. Task 1 — State Reconstruction (35% weight)

### Strategy: Spatiotemporal LightGBM with Flow-Balance Awareness

The baseline (historical mean) scores ~0.67 total. Current LB leader uses
spatiotemporal LightGBM with contextual features and scores 0.845. You beat
that by adding flow-balance regularization.

### Step 2.1 — Build the feature matrix

For each masked cell (corridor, link, timestamp):

```
Target:     speed_kmh, flow_vph (two separate models per corridor)

Features:
  Time features:
    - hour_of_day, day_of_week, is_weekend, month
    - sin/cos encoding of hour_of_day (captures periodicity)

  Same-link temporal context (history, no leakage):
    - speed/flow at T-5, T-10, T-15, T-30, T-60 from UNMASKED cells
    - speed/flow at same time yesterday, 1 week ago (from train)
    - historical mean at (link, hour, dow) computed from train only
    - exponentially weighted recent mean (α=0.3) from available cells

  Spatial neighbor context (same timestamp):
    - upstream link speed/flow (if observed)
    - downstream link speed/flow (if observed)
    - on-ramp flow at same link (treat null as NaN, not 0)
    - off-ramp flow at same link

  Derived features:
    - density = flow / (speed × n_lanes) for observed neighbors
    - occupancy from observed neighbors
    - regime flag: R1 (20% masked), R2 (30%), R3 (50%) — day-level

  Network features (static, from corridor config):
    - link_length, n_lanes, free_flow_speed, link_position_in_corridor
```

**IMPORTANT:** Separate models per corridor OR per corridor+regime. The 10
corridors have different geometries — pooling all 10 without corridor embeddings
hurts. Train one LightGBM per corridor (10 models for speed, 10 for flow = 20
models). This is feasible; each corridor has ~273 training days × 288 timesteps
× ~15 links = ~1.2M rows.

### Step 2.2 — Flow-Balance-Aware Post-Processing

This is what pushes Task 1 AND Task 3 simultaneously.

After LightGBM predictions, run a flow-balance projection:

```python
# For each 5-minute interval, for each link:
# Conservation: N(t+dt) - N(t) = dt * (q_in + r_on - q_out - r_off)
# Where: N = k * link_length, k = flow / speed

# Step 1: Accept all observed (non-masked) cells as fixed
# Step 2: For masked cells, solve a constrained least-squares:
#   minimize ||v_pred - v_lgbm||^2 + ||q_pred - q_lgbm||^2
#   subject to:
#     conservation at each link (using available ramp flows)
#     fundamental diagram: k = q/v, flow <= capacity
#     non-negativity: q >= 0, v > 0
# Use scipy.optimize.minimize with SLSQP or scipy.optimize.lsq_linear
```

Even a soft version (penalize conservation violations rather than hard
constraints) will lift Task 3 substantially without hurting Task 1 accuracy.

### Step 2.3 — Training procedure

```python
import lightgbm as lgb

params = {
    'objective': 'regression_l1',   # MAE → more robust to outliers
    'n_estimators': 2000,
    'learning_rate': 0.05,
    'num_leaves': 127,
    'min_child_samples': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_jobs': -1,
}
# Separate speed and flow models
# Use early stopping on 20% held-out validation days from train split
# Store OOF predictions for post-processing calibration
```

### Step 2.4 — Expected Task 1 gain from this approach

| Approach | Approx S_state |
|----------|---------------|
| Historical mean baseline | ~0.72 |
| Simple time interpolation | ~0.79 |
| Per-corridor spatiotemporal LightGBM | ~0.87 |
| + Flow-balance post-processing | ~0.91+ |

---

## 3. Task 2 — Queue Forecasting (30% weight)

### Strategy: Per-Window-Type LightGBM with Onset Exploitation

The baseline (persistence) scores 0.2518. LB leader scores ~0.93 onset IoU
using the onset regularization trick.

### Step 3.1 — Understanding the window structure

Each window gives 60 minutes of history (12 timesteps) up to T, asks about
T+5 through T+30 (6 steps). Two types:
- `queue_onset`: no queue in history, queue appears in horizon
- `queue_ongoing`: queue visible at T, continues through horizon

A link is "queued" when speed ≤ 0.60 × free_flow_speed.

**The onset trick:** In 39/40 training onset windows, the first queued cell
appears at T+30 exactly (not T+5 or T+15). This means:
- For onset windows: predict T+30 as the dominant queued timestep
- The model learns this naturally if trained on windows constructed the same way
- Build training windows by sampling ONLY days where congestion onset starts at
  T+30 in the label horizon

### Step 3.2 — Feature engineering

```
Per-link per-window features:
  - Current speed at T, T-5, T-10, ..., T-55 (12 historical values)
  - Current flow at T, T-5, ..., T-55
  - Queue flag at T (is speed ≤ 0.6 × ffs at each link at T)
  - Distance downstream from nearest active queue at T
  - Distance upstream from nearest ramp at T
  - Shockwave travel time estimate: length / (ffs - current_speed)
  - Hour of day, day of week (affects congestion onset timing)
  - Free-flow speed for this link (from config)
  - Window type (onset=0, ongoing=1) — train separate models

Per-horizon-step output:
  - Binary: is link queued at T+h for h in [5,10,15,20,25,30]
```

### Step 3.3 — Model architecture

Train **separate LightGBM classifiers** for onset vs ongoing windows:

```python
# For onset windows (the high-value case):
onset_params = {
    'objective': 'binary',
    'metric': 'auc',
    'n_estimators': 500,
    'num_leaves': 63,
    'learning_rate': 0.05,
}
# Threshold tuning: optimize IoU directly on OOF predictions
# IoU ≠ F1. Tune threshold on OOF by grid search over [0.1, 0.9]

# For ongoing windows:
# Queue propagates upstream at ~shockwave speed
# Feature: time_since_onset at T for upstream links
# These are easier — persistence baseline is already decent here
```

### Step 3.4 — Causal constraint (critical)

**NEVER use features from after T.** This seems obvious but is easy to break
when building day-level features. Add a unit test:

```python
def test_no_leakage(features_df):
    # Ensure no column uses data from T+1 through T+30
    # Overwrite the horizon slots with NaN and recompute features
    # Predictions must be identical
    assert no difference in predictions
```

### Step 3.5 — IoU threshold calibration

The scoring is IoU (intersection over union), not accuracy. At decision threshold θ:
```python
from sklearn.metrics import jaccard_score  # IoU = Jaccard

# Grid search threshold per window type:
for threshold in np.arange(0.1, 0.9, 0.02):
    pred_binary = (proba_predictions > threshold).astype(int)
    iou = jaccard_score(true_labels_flat, pred_binary_flat)
    # Pick threshold that maximizes IOU on OOF
```

---

## 4. Task 3 — Physics Compliance (15% weight)

### You submit nothing new for Task 3. It re-reads your Task 1 file.

The only lever is Task 1 quality. The two rules:

**Fundamental Diagram (S_FD):**
- Don't submit speeds and flows that are physically impossible together
- k = q/v must be ≤ jam density (from corridor config)
- At high density, flow must be below the parabola
- Clip predictions to physical bounds BEFORE submitting Task 1

```python
# After LightGBM prediction, enforce FD bounds per link:
jam_density = corridor_config['jam_density']
capacity_flow = corridor_config['capacity']
free_flow_speed = corridor_config['free_flow_speed']

# Triangular fundamental diagram:
# q = min(k * v_f, capacity - (k - k_c) * w)
# Clip flow predictions to [0, capacity]
# Clip speed to [0, free_flow_speed]
# Don't submit zero flow at zero density (flags as empty road penalty)
```

**Conservation (S_LWR) — the big one (2/3 weight in Task 3):**
- This is where the flow-balance post-processing from Task 1 pays off
- The projection step described in Task 1 directly optimizes S_LWR
- Even partial conservation improvement moves the score significantly

**Expected Task 3 scores:**
| Approach | S_physics |
|----------|-----------|
| Baseline / any simple imputation | ~0.32 |
| FD-aware clipping only | ~0.60 |
| FD + flow-balance projection | ~0.85+ |

---

## 5. Task 4 — OD Matrix Estimation (20% weight)

### Strategy: NNLS with per-split counts + deviation penalty

The scoring breakdown:
```
S_ODME = 0.45 × S_od + 0.25 × S_link + 0.15 × S_dev + 0.15 × S_attr
```

**Key insight from LB leader:** Use each split's own released link counts,
not the training counts in the organizer's baseline script. The released counts
are noise-free → unregularized NNLS fits them exactly → S_link = 1.0.

### Step 5.1 — Formulation

```
Given:   A (path-link incidence matrix, P paths × L links)
         c (observed link counts, L × 1)
         b (prior path flows, P × 1)

Solve:   minimize ||Ax - c||^2 + λ||x - b||^2
         subject to: x >= 0

This is a regularized NNLS:
  Augmented system: [A; sqrt(λ)*I] x = [c; sqrt(λ)*b]
  Solve with scipy.optimize.nnls or sklearn NNLS
```

### Step 5.2 — Hyperparameter: λ (regularization strength)

The `S_dev` term penalizes moving too far from the prior, which blocks two
degenerate solutions:
- λ = 0: fits counts perfectly but may wildly deviate from prior
- λ → ∞: returns the prior but ignores counts

Grid search λ over [0.001, 0.01, 0.05, 0.1, 0.5] on the train split.
Since Task 4 ground truth is available for train, you can score yourself.

```python
from scipy.optimize import nnls
import numpy as np

def solve_odme(A, c, b, lam=0.05):
    n_paths = A.shape[1]
    # Augment the system
    A_aug = np.vstack([A, np.sqrt(lam) * np.eye(n_paths)])
    c_aug = np.concatenate([c, np.sqrt(lam) * b])
    x, residual = nnls(A_aug, c_aug)
    return x
```

### Step 5.3 — S_attr optimization

`S_attr` checks whether your destination-attraction distribution matches the
reference. This is a marginal distribution over paths grouped by destination.
Ensure your path flows, when summed by destination, match the prior's
destination-level distribution. Post-process by rescaling path flows within
each destination group to match prior totals if needed.

### Step 5.4 — Expected Task 4 score

| Approach | S_ODME |
|----------|--------|
| Organizer's baseline (train counts) | ~0.70 |
| NNLS on split-specific counts (λ=0.05) | ~0.82 |
| + λ tuned per corridor + attr rescaling | ~0.88+ |

---

## 6. Submission Assembly

```python
# Use the official merge script:
python src/merge_submissions.py \
  --state   state_submission.csv \
  --queue   queue_submission.csv \
  --odme    odme_submission.csv \
  --output  submission.csv

# Final row count should be ~6,980,503 rows (matches sample_submission.csv)
# Validate schema against sample_submission.csv before uploading
```

---

## 7. Execution Timeline (3 weeks to deadline)

### Week 1: Foundation (Days 1–7)

| Day | Task |
|-----|------|
| 1 | Clone official repo, install deps, run all baselines, confirm local scores |
| 2 | EDA: plot speed/flow heatmaps per corridor, understand R1/R2/R3 patterns, find congestion onset patterns |
| 3–4 | Task 4: implement per-split NNLS, tune λ, score on train, first submission |
| 5–6 | Task 1: build feature engineering pipeline, train LightGBM (historical mean + time features) |
| 7 | First full submission: LightGBM Task 1 + persistence Task 2 + Task 3 auto + improved Task 4 |

**Target end of Week 1:** ~0.80 total

### Week 2: High-Value Improvements (Days 8–14)

| Day | Task |
|-----|------|
| 8–9 | Task 2: build per-window-type LightGBM, implement onset trick, causal unit tests |
| 10 | Task 2: IoU threshold calibration, submission, compare vs baseline |
| 11–12 | Task 1: add spatial neighbor features, flow-balance post-processing (soft constraint) |
| 13 | Task 3: FD clipping, validate physics scores improve on LB |
| 14 | Second full submission, analyze per-corridor LB breakdown |

**Target end of Week 2:** ~0.87 total

### Week 3: Precision Tuning (Days 15–21)

| Day | Task |
|-----|------|
| 15–16 | Task 1: hard flow-balance projection (constrained NNLS per timestep) |
| 17 | Task 2: add shockwave propagation features, re-tune threshold per corridor |
| 18 | Task 4: per-corridor λ, S_attr rescaling |
| 19–20 | Ensemble: stack predictions from 2–3 Task 1 model variants |
| 21 | Final validation check, clean up submission, submit |

**Target end of Week 3:** >0.90 total

---

## 8. Corridor-Specific Notes

The 10 corridors are:
- `D7_I10_E/W`, `D7_I210_E/W`, `D7_I405_N/S`
- `D12_I5_N/S`, `D12_I405_N/S`

Note: `D12_I405_N` and `D12_I405_S` are **excluded from Task 2 only** but
scored normally on all other tasks. Do not accidentally omit them from Tasks
1/3/4.

Each corridor has its own fundamental diagram parameters (free_flow_speed,
capacity, jam_density) in the config. NEVER use another corridor's params.

---

## 9. Validation Strategy

Since private split ≠ validation split, avoid public-LB overfitting:

```
train (273 days)
  → hold out last 30 days as local "validation"
  → train on days 1-243
  → score locally on days 244-273

This approximates the private split scenario (one month of held-out data).
Track:
  - S_state (local, all three regimes separately)
  - S_queue (cannot score locally — use LB sparingly)
  - S_ODME (local, full scoring available)
```

Submit to public LB no more than once every 3 days to preserve submission
quota and avoid overfitting to the validation month.

---

## 10. What Will Beat the Current LB Leader (0.845)

The LB leader's reported approach:
- Task 1: spatiotemporal LightGBM (~56M training rows)
- Task 2: per-window-type LightGBM + onset trick → IoU 0.93
- Task 3: passive (came from Task 1)
- Task 4: NNLS on split-specific counts

**Your edge over them:**

| Area | Their approach | Your advantage |
|------|---------------|----------------|
| Task 1 | Spatiotemporal LightGBM | + flow-balance projection for conservation |
| Task 2 | Already strong (onset trick) | Match + per-corridor threshold tuning |
| Task 3 | Passive | + FD clipping + conservation projection → S_LWR gain |
| Task 4 | NNLS on split counts | + per-corridor λ tuning + S_attr rescaling |

The flow-balance projection on Task 1 is the single biggest differentiator.
If Task 3 S_LWR goes from 0.70 → 0.90, that alone adds 0.15 × (2/3) × 0.20 = ~0.02
to total score. Combined with Task 1 reconstruction quality improvements, a
total score above 0.90 is achievable.

---

## 11. Code Skeleton for Gemini 3.8 CLI

When passing this plan to Gemini CLI, also provide these specific code prompts:

**Prompt 1 (Task 1 features):**
> "Build a feature engineering function `build_state_features(corridor_df, config, masked_cells)` that constructs the feature matrix described in Section 2.1. Use Polars for efficiency. Output a DataFrame with columns: [submission_id, speed_target, flow_target, + all features]. Handle null ramp flows as NaN."

**Prompt 2 (Task 1 model training):**
> "Train per-corridor LightGBM models for speed and flow prediction using the feature matrix. Use train split days 1-243, validate on days 244-273. Return OOF predictions and final models. Use params from Section 2.3."

**Prompt 3 (Flow balance post-processing):**
> "Implement `flow_balance_projection(state_df, ramp_df, config)` that applies soft conservation constraints to Task 1 LightGBM predictions. Use SLSQP optimization per 5-minute timestep. Penalize |N(t+dt) - N(t) - dt*(q_in + r_on - q_out - r_off)| in the objective."

**Prompt 4 (Task 2):**
> "Build Task 2 LightGBM models for queue_onset and queue_ongoing windows separately. Features per Section 3.2. Train on windows where first queue appears at T+30 for onset type. Include IoU threshold calibration on OOF. Add causal unit test."

**Prompt 5 (Task 4):**
> "Implement `solve_odme(A, c, b, lam)` using scipy NNLS on per-split link counts. Grid-search λ over [0.001, 0.01, 0.05, 0.1, 0.5] on train split. Add S_attr rescaling post-processing."

**Prompt 6 (Submission assembly):**
> "Use the official `src/merge_submissions.py` to combine task outputs. Validate against sample_submission.csv schema. Check row count matches."

---

## 12. Expected Final Score Breakdown

| Task | Weight | Baseline | This Plan |
|------|--------|----------|-----------|
| Task 1 (State) | 35% | 0.72 | 0.91 |
| Task 2 (Queue) | 30% | 0.25 | 0.90 |
| Task 3 (Physics) | 15% | 0.32 | 0.85 |
| Task 4 (ODME) | 20% | 0.70 | 0.87 |
| **Total** | **100%** | **0.55** | **~0.895** |

With strong execution of flow-balance projection, pushing Task 3 to 0.90+
gets the total above 0.91.
