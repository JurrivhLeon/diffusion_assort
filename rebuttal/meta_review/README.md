# Meta-Review Response

This directory contains the tracked code, configurations, result summaries, and response text used to address the meta-review. Raw reviews, the submitted manuscript, private notes, checkpoints, datasets, and large outputs remain in ignored local directories.

## Response priorities

1. Isolate the contribution of diffusion by comparing D3AO with MIP, local search, simulated annealing, and random search under the same frozen GAsN/RAsN reward model.
2. Correct the asymmetric best-of-256 protocol by selecting the returned D3AO assortment using estimated reward and using ground-truth reward only for post-selection evaluation.
3. Match optimization methods by both reward-query budget and wall-clock budget.
4. Add a cardinality-constrained experiment with the same feasible set for every method.
5. Add an experiment beyond \(N=100\), reporting quality, runtime, query count, and solver status.
6. Evaluate sensitivity to logging-policy quality and coverage.
7. Ablate the reverse-step count, accelerated update, guidance schedule, guidance strength, and candidate count.
8. Add real-data evidence when a defensible offered-assortment/choice dataset is available; otherwise add a clearly labeled out-of-family synthetic stress test and narrow the corresponding claims.

## Planned structure

- `configs/`: version-controlled experiment configurations.
- `scripts/`: rebuttal-specific launch and aggregation scripts.
- `results/`: compact, version-controlled summaries; large raw outputs remain ignored.
- `response.md`: concise response draft populated only with verified results.

## Neural-reward MIP suite

The tracked configuration `configs/neural_mip_suite.json` reproduces the submitted synthetic grid by default. The runner supports smaller overrides for pilots and can resume each stage independently.

Preview a single case without running it:

```bash
python rebuttal/meta_review/scripts/run_neural_mip_suite.py \
  --choice-models mnl \
  --n-products 20 \
  --betas 1.0 \
  --seeds 0 \
  --stages generate train mip \
  --dry-run
```

Run data generation and matched GAsN/RAsN training:

```bash
python rebuttal/meta_review/scripts/run_neural_mip_suite.py \
  --choice-models mnl \
  --n-products 20 \
  --betas 1.0 \
  --seeds 0 \
  --stages generate train
```

Run the two MIP solves in a Gurobi-enabled environment:

```bash
python rebuttal/meta_review/scripts/run_neural_mip_suite.py \
  --choice-models mnl \
  --n-products 20 \
  --betas 1.0 \
  --seeds 0 \
  --stages mip
```

The lower-level Python runner defaults to repository-local ignored data/output directories when invoked directly. The formal SLURM launcher overrides these locations as described below.

## Formal MIP-versus-D3AO experiment

The formal launcher defaults to the main MMNL misspecification setting with \(N=100\), both submitted behavior-policy temperatures, 10 dataset seeds, and 10 D3AO sampling seeds:

```bash
sbatch rebuttal/meta_review/scripts/run_formal_mip_vs_d3ao.sh
```

The tracked script requests the `biostat` partition/account, 20 CPU cores, 200 GB memory, and a 20-hour wall-clock limit per task. It uses the array specification `0-19%5`: the default 20 combinations of two behavior-policy temperatures and ten dataset seeds are split into independent jobs, with at most five running concurrently. Each task runs both reward models and all configured D3AO sampling seeds for one dataset. SLURM stdout/stderr are written under `/hpc/group/xielab/xj58/sbatch_output/`, and job notifications are sent to `xj58@duke.edu`.

The job initializes Conda explicitly and activates the `SpaRest` environment before running any command. Set `CONDA_ENV` at submission time only if a different environment is required. The job checks PyTorch immediately and, when `RUN_MIP=1`, also checks `gurobipy` before starting data generation or training.

Formal experiment artifacts are stored outside the repository under:

```text
/hpc/home/xj58/da_rebuttal/
├── datasets/   # Generated logged assortment-choice JSON datasets
└── outputs/    # Checkpoints, manifests, logs, MIP results, and D3AO samples
```

The launcher creates both directories automatically. Override the common root with `ARTIFACT_ROOT`, or override `DATA_ROOT` and `OUTPUT_ROOT` separately when needed.

Its primary comparison is GAsN-reward MIP versus GAsN-guided D3AO, and RAsN-reward MIP versus RAsN-guided D3AO. Each D3AO run generates 256 candidates and returns the candidate selected by fitted reward. Ground-truth reward is used only for post-selection evaluation.

Environment variables can reduce or extend the grid without editing the script:

```bash
sbatch --export=ALL,CHOICE_MODELS=mnl,N_PRODUCTS=20,BETAS=1.0,DATA_SEEDS=0,SAMPLING_SEEDS=0,N_SAMPLES=1000,DEVICE=cpu,RUN_MIP=0 \
  rebuttal/meta_review/scripts/run_formal_mip_vs_d3ao.sh
```

The static SBATCH array contains 20 task IDs. If an override defines fewer than 20 data configurations, extra tasks exit successfully without doing work. If an override defines more than 20 configurations, also override the array range at submission time. For example, a 30-case grid requires `sbatch --array=0-29%5 ...`.

Useful overrides are `PYTHON_BIN`, `CONDA_ENV`, `CHOICE_MODELS`, `N_PRODUCTS`, `BETAS`, `DATA_SEEDS`, `SAMPLING_SEEDS`, `N_SAMPLES`, `DEVICE`, `NUM_CANDIDATES`, `GUIDANCE_SCALE`, `GUIDANCE_GAMMA`, `RUN_MIP`, `FORCE`, `ARTIFACT_ROOT`, `DATA_ROOT`, and `OUTPUT_ROOT`.

## Evaluation rules

- Freeze the learned GAsN/RAsN checkpoint when comparing optimizers.
- Use the same plug-in reward scorer for all optimization methods.
- Select deployable candidates using estimated reward only.
- Use ground-truth reward only for post-selection evaluation in synthetic experiments.
- Label oracle best-of-sample statistics as diagnostics rather than returned solutions.
- Report all timeouts, solver statuses, random seeds, query counts, and feasibility violations.
- Do not claim exact MIP optimization when piecewise approximations or nonzero solver gaps are used.

## Status

- [ ] Recover and verify the submitted experiment configurations.
- [ ] Train or recover matched GAsN and RAsN checkpoints.
- [ ] Implement the shared optimization-budget interface.
- [ ] Run the optimizer-isolation experiment.
- [ ] Correct and rerun candidate selection.
- [ ] Run the cardinality-constrained experiment.
- [ ] Run the larger-scale experiment.
- [ ] Run logging-coverage sensitivity experiments.
- [ ] Run reverse-process and candidate-count ablations.
- [ ] Complete the real-data or out-of-family study.
- [ ] Draft and verify the final meta-review response.
