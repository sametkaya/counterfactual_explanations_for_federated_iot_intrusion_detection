#!/usr/bin/env bash
# Optional diagnostics (not used in the manuscript): the beta2=0.9 ablation on the
# normalisation-free network and the 1D-CNN (A and C, 3 seeds), and the Adam step-factor trace on
# the label-smoothing and input-noise detectors.
# Sequential; stops at the first failure. Usage: bash run_optional_diagnostics.sh
set -e
export JE_DEVICE=cpu
cd "$(dirname "$0")"
echo "JE_DEVICE=$JE_DEVICE  start $(date)"

# --- code smoke test (minutes) ---
python3 -m journal_extension.exp19_controls --dataset ciciot2023 --quick --parts smooth,diag --smooth-arms E_LabelSmooth --diag-variants adam --diag-samples 5 --out-suffix _smoke4

# --- Adam second-moment ablation on the harder baselines (A and C, 3 seeds, 100 queries) ---
python3 -m journal_extension.exp19_controls --dataset ciciot2023 --max-samples 200000 --batch-size 256 --n-runs 3 --cf-samples 100 --cf-max-iter 300 --arch nonorm --parts diag --diag-variants adam,adam_b2_0.9 --diag-samples 100 --diag-conds A_Standard,C_Noise_Only --out-suffix _diag_nonorm
python3 -m journal_extension.exp19_controls --dataset ciciot2023 --max-samples 200000 --batch-size 256 --n-runs 3 --cf-samples 100 --cf-max-iter 300 --arch cnn --parts diag --diag-variants adam,adam_b2_0.9 --diag-samples 100 --diag-conds A_Standard,C_Noise_Only --out-suffix _diag_cnn

# --- step-factor trace on the smoothing detectors (label smoothing, input noise; 3 seeds) ---
python3 -m journal_extension.exp19_controls --dataset ciciot2023 --max-samples 200000 --batch-size 256 --n-runs 3 --cf-samples 100 --cf-max-iter 300 --parts smooth,diag --smooth-arms E_LabelSmooth,F_InputNoise --diag-variants adam --diag-samples 100 --out-suffix _smoothdiag

echo "all done $(date)"
