$ErrorActionPreference = "Stop"

if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    & ".\.venv\Scripts\Activate.ps1"
}

$REL = "./data/kaggle_public"

Write-Host "=== Task 4: ODME ==="
python src/task4/solve_odme.py --release-root $REL --split all

Write-Host "=== Task 1: State Reconstruction ==="
python src/task1/train_and_predict.py --release-root $REL --split all

Write-Host "=== Task 2: Queue Forecasting ==="
python src/task2/train_and_predict.py --release-root $REL --split all

Write-Host "=== Merge Submissions ==="
python official_toolkit/src/merge_submissions.py `
  --state  outputs/submissions/state_submission.csv `
  --queue  outputs/submissions/queue_submission.csv `
  --odme   outputs/submissions/odme_submission.csv `
  --key    "$REL/submission_key.csv" `
  --output outputs/submissions/submission_final.csv

Write-Host "Done. Upload outputs/submissions/submission_final.csv to Kaggle."
