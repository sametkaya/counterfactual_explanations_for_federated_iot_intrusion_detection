#!/usr/bin/env bash
# Membership audits and optimiser diagnostics: offline LiRA under the main recipe, at 32k/16k
# records with 100 epochs and in the memorisation regime, the canary positive control (with and
# without exact-duplicate removal) and the Adam-moment diagnostics of the generator.
# Sequential; stops at the first failure. Usage: bash run_audit_diagnostics.sh
set -e
export JE_DEVICE=cpu
cd "$(dirname "$0")"
echo "JE_DEVICE=$JE_DEVICE  start $(date)"

# --- code smoke tests (minutes) ---
python3 -m journal_extension.exp15_lira_mia --dataset ciciot2023 --quick --canaries 20 --dedup --no-dropout --batch-size 32 --out-suffix _smoke2
python3 -m journal_extension.exp19_controls --dataset ciciot2023 --quick --parts diag --diag-samples 5 --out-suffix _smoke3

# --- LiRA, main recipe (160k records, 20 epochs, batch 256) ---
python3 -m journal_extension.exp15_lira_mia --dataset ciciot2023 --max-samples 200000 --n-shadow 32 --n-audit 2000 --shadow-train-size 40000 --shadow-epochs 20 --epochs 20 --batch-size 256 --out-suffix _v2
python3 -m journal_extension.exp15_lira_mia --dataset edge-iiot --max-samples 200000 --n-shadow 32 --n-audit 2000 --shadow-train-size 40000 --shadow-epochs 20 --epochs 20 --batch-size 256 --out-suffix _v2

# --- LiRA, 32k/16k records, 100 epochs ---
python3 -m journal_extension.exp15_lira_mia --dataset ciciot2023 --max-samples 40000 --n-shadow 32 --n-audit 2000 --shadow-train-size 8000 --shadow-epochs 100 --epochs 100 --batch-size 256 --out-suffix _mem3
python3 -m journal_extension.exp15_lira_mia --dataset edge-iiot --max-samples 20000 --n-shadow 32 --n-audit 2000 --shadow-train-size 8000 --shadow-epochs 100 --epochs 100 --batch-size 256 --out-suffix _mem3

# --- LiRA positive control: 2k records, 300 epochs, batch 32, no dropout, 200 canaries ---
python3 -m journal_extension.exp15_lira_mia --dataset ciciot2023 --max-samples 40000 --n-shadow 32 --n-audit 1000 --shadow-train-size 2000 --shadow-epochs 300 --epochs 300 --target-train-size 2000 --batch-size 32 --no-dropout --canaries 200 --out-suffix _canary
python3 -m journal_extension.exp15_lira_mia --dataset edge-iiot --max-samples 40000 --n-shadow 32 --n-audit 1000 --shadow-train-size 2000 --shadow-epochs 300 --epochs 300 --target-train-size 2000 --batch-size 32 --no-dropout --canaries 200 --out-suffix _canary

# --- same, exact duplicates removed before the split ---
python3 -m journal_extension.exp15_lira_mia --dataset ciciot2023 --max-samples 40000 --n-shadow 32 --n-audit 1000 --shadow-train-size 2000 --shadow-epochs 300 --epochs 300 --target-train-size 2000 --batch-size 32 --no-dropout --canaries 200 --dedup --out-suffix _canary_dedup
python3 -m journal_extension.exp15_lira_mia --dataset edge-iiot --max-samples 40000 --n-shadow 32 --n-audit 1000 --shadow-train-size 2000 --shadow-epochs 300 --epochs 300 --target-train-size 2000 --batch-size 32 --no-dropout --canaries 200 --dedup --out-suffix _canary_dedup

# --- Adam-moment ablation and optimiser diagnostics (A and C, 3 seeds, 100 queries) ---
python3 -m journal_extension.exp19_controls --dataset ciciot2023 --max-samples 200000 --batch-size 256 --n-runs 3 --cf-samples 100 --cf-max-iter 300 --parts diag --diag-variants adam,adam_b1_0,adam_b2_0.9,signgd --diag-samples 100 --diag-conds A_Standard,C_Noise_Only --out-suffix _diag
python3 -m journal_extension.exp19_controls --dataset edge-iiot --max-samples 200000 --batch-size 256 --n-runs 3 --cf-samples 100 --cf-max-iter 300 --parts diag --diag-variants adam,adam_b1_0,adam_b2_0.9,signgd --diag-samples 100 --diag-conds A_Standard,C_Noise_Only --out-suffix _diag

echo "all done $(date)"
