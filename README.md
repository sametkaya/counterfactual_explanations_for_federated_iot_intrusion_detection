# Counterfactual explanations for federated IoT intrusion detection

Code, logs and per-run results for two papers:

1. **S. Kaya, "Federated learning-based counterfactual explanations for resolving the
   privacy-explainability dilemma in IoT intrusion detection", SIU 2026**
   (DOI 10.1109/SIU71813.2026.11636543, in Turkish). Protocol P0: `run_comprehensive_experiments.py`,
   results in `experiments/`.
2. **S. Kaya, "Privacy noise reshapes the decision boundary of IoT intrusion detectors: a rise in
   counterfactual validity is not evidence of better explanations"** (journal manuscript).
   Protocol P1: `journal_extension/`, results in `experiments_journal/`, table and figure
   scripts in `journal_paper/`.

The manuscripts themselves are not part of the repository.

## Layout

```
src/                      detector (models/), DP-SGD-style training loop and the sparsity-capped
                          counterfactual generator (counterfactual/), data loaders (data/),
                          federated components (federated/), evaluation helpers (evaluation/)
run_comprehensive_experiments.py, visualization.py
                          protocol P0 (SIU 2026): five-run sweeps, federated aggregation, figures
experiments/              P0 results: exp_02..exp_07 (TON_IoT, CICIoT2023, Edge-IIoTset), cross-dataset figures
journal_extension/        protocol P1 (journal): common.py (data cache, training, generator wrappers),
                          exp12b/13b (noise sweeps and factorials, n=10), exp14 (boundary geometry),
                          exp15 (offline LiRA with canary control), exp16 (feasibility-aware actionability),
                          exp17 (architecture check), exp18 (FedAvg replication), exp19 (controls,
                          generator variants, budget curves, optimiser diagnostics, smoothing arms),
                          exp20 (certified DP-SGD with Opacus); run_journal_experiments.py
experiments_journal/      P1 results: one folder per run, each with results.json and log.txt
journal_paper/            scripts that turn experiments_journal/ into the manuscript's tables and figures
run_audit_diagnostics.sh  membership audits, canary control and optimiser diagnostics (complete command lines)
run_optional_diagnostics.sh
                          optional diagnostics not used in the manuscript
```

## Environment

Python 3.10+, PyTorch 2.x (CPU is sufficient; the journal runs were made on CPU with `JE_DEVICE=cpu`),
`pip install -r requirements.txt`. The certified arm (exp20) needs `opacus`.

## Datasets

Place the raw CSV files under `dataset/` (git-ignored):

```
dataset/CICIoT2023/   the 63 merged CSV files
dataset/Edge-IIoTset/ DNN-EdgeIIoT-dataset.csv
dataset/TON_IoT/      Train_Test_Network.csv
```

`journal_extension/common.py` reads the raw files once and caches the standardised sample under
`dataset/_cache/` (CICIoT2023 takes several minutes on the first run); a change in preprocessing
invalidates the cache automatically.

## Reproducing the journal experiments

Every run writes `experiments_journal/<experiment>_<suffix>_<dataset>/{results.json,log.txt}`.
Seeds are fixed in `journal_extension/common.py` (42 upward); CPU runs are deterministic up to
library versions.

```
export JE_DEVICE=cpu
# noise sweeps and 2x2 factorials, ten seeds (Tables 3-4)
python3 -m journal_extension.exp12b_13b_rerun --dataset ciciot2023 --max-samples 200000 --batch-size 256
python3 -m journal_extension.exp12b_13b_rerun --dataset edge-iiot  --max-samples 200000 --batch-size 256
# boundary geometry (Section 6.2)
python3 -m journal_extension.exp14_mechanism --dataset ciciot2023 --max-samples 200000 --batch-size 256
# controls (Section 6.3): core set, then the variants, curves and diagnostics
python3 -m journal_extension.exp19_controls --dataset ciciot2023 --max-samples 200000 --batch-size 256 --n-runs 3 --parts core,hist2,variants,sparsedf --out-suffix _mech
python3 -m journal_extension.exp19_controls --dataset ciciot2023 --max-samples 200000 --batch-size 256 --n-runs 10 --parts hist,discrete,deepfool,hist2 --out-suffix _k10
bash run_audit_diagnostics.sh      # membership audits, canary control, optimiser diagnostics (Tables 7, 11, 12)
bash run_optional_diagnostics.sh   # optional: beta2 ablation on the harder baselines, smoothing-arm traces
# certified arm (Section 6.5), actionability (Section 6.4), architecture and federated checks (Appendix)
python3 -m journal_extension.exp20_opacus --dataset ciciot2023 --max-samples 200000 --batch-size 256 --arch nonorm --n-runs 10 --out-suffix _nonorm10
python3 -m journal_extension.exp16_domain_cf --dataset ciciot2023 --max-samples 200000 --batch-size 256 --n-runs 5 --model-conditions A_Standard,B_DP_SGD --out-suffix _ab_cpu
python3 -m journal_extension.exp17_architecture --dataset ciciot2023 --max-samples 200000 --batch-size 256
python3 -m journal_extension.exp18_federated --dataset ciciot2023 --max-samples 200000 --batch-size 256
```

The flags of every run that appears in the manuscript are recorded in the first lines of its
`log.txt` (the exact spelling of the condition flags is in each script's `--help`); `run_audit_diagnostics.sh` holds the complete command lines of the audit and diagnostic runs.

## Tables and figures

```
python3 journal_paper/make_tables_exp19.py   # controls, generator variants, histograms
python3 journal_paper/make_tables_regime.py  # regime table, optimiser variants, actionability, ten-seed certified arm
python3 journal_paper/make_table_lira.py     # membership audit (fixed-variance LiRA re-scored from the stored raw statistics)
python3 journal_paper/make_table_diag.py     # optimiser diagnostics
python3 journal_paper/make_table_smooth.py   # smoothing interventions
python3 journal_paper/make_fig_curves.py     # budget curves
python3 journal_paper/make_figures.py        # the remaining figures
python3 journal_paper/stats_tables.py        # paired statistics of the sweeps and factorials
```

All scripts take the results folder as an optional first argument (default `experiments_journal`).

## Reproducing the SIU 2026 experiments (protocol P0)

```
python3 run_comprehensive_experiments.py --dataset ciciot2023   # likewise edge-iiot, toniot
python3 visualization.py
```

Results and figures are written to `experiments/exp_0x_<dataset>/`.

## Licence

GNU GPL v3, see `LICENSE`.
