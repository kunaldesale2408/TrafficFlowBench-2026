# TrafficFlowBench — 0.86517 solution assets

Checkpoints, feature code and training scripts for submission v20 (public score 0.86517, reference 56843335). It ranked 48/320 in the October 5, 2026 snapshot. Only released competition observations, metadata, counts and weak priors are used.

## Inference

Attach this dataset and the competition data to the companion notebook. Use a GPU session for CatBoost state inference; CPU fallback is supported but slower. The notebook generates predictions from scratch and solves ODME. No ready-made final submission is included. Numerical results can vary across library/GPU versions.

## Retraining

Install numpy, pandas, scipy, pyarrow, LightGBM and CatBoost. Set TFB_DATA_ROOT to the official kaggle_public directory, TFB_OUTPUT_ROOT to a writable training directory, and optionally TFB_THREADS (default 4). Run the following from the source directory:

```bash
python prepare_state.py
python train_state.py --channel speed --device 0 --depth 8 --iterations 9000 --tag d8
python train_state.py --channel flow --device 0 --depth 8 --iterations 9000 --tag d8
python train_state.py --channel speed --device 0 --depth 10 --iterations 4500 --tag d10
python train_state.py --channel flow --device 0 --depth 10 --iterations 4500 --tag d10
python train_ongoing.py
python train_onset.py
python train_onset_catboost.py
```

The state arrays contain 3.675 million training rows and 420,000 February validation rows. Feature preparation accumulates arrays in RAM and can exceed small notebook limits; original training used a workstation with ample RAM and four GPUs. The four state training jobs can be run sequentially on one GPU. Their saved predictions are averaged 50/50 by depth for each target, as recorded in state_settings.json. Copy that JSON beside any newly trained models when using the inference class. GPU retraining is not bitwise deterministic.

train_onset.py reproduces the two-stage LightGBM onset model. train_onset_catboost.py then trains the augmented CatBoost model and evaluates the blend/threshold grid. It expects onset_lgb.txt in TFB_OUTPUT_ROOT. TFB_GPU_DEVICE defaults to 0. The published inference blend is 50% LightGBM and 50% CatBoost, threshold 0.55. Ongoing windows retain the original 36-feature LightGBM at threshold 0.40.

state_model.py supplies the original 54-feature interpolation/spatial baseline. prepare_state.py adds 40 lag/lead quality-context features and builds two independent masking realizations per training month. Its old LightGBM training helper is retained as baseline source, but the four state models used by this solution are CatBoost checkpoints.

odme_initial.parquet is a saved initial solution derived from released counts/prior, not hidden truth. The notebook exposes both the original initialization/600-iteration procedure and the tighter 8,000-iteration continuation. The full ODME score is unavailable locally; included diagnostics measure count fit and numerical convergence.

Local validation tables and the original package versions are included. The public score is the historical submission result; private performance remains unknown.


## Public notebook

Run the inference notebook at https://www.kaggle.com/code/anubisyy/trafficflowbench-top-50-0-86517-code . Its default run uses these original checkpoints and writes submission.csv. The score and rank are dated historical results.
