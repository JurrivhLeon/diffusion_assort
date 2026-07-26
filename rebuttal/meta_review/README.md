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

The formal workflow is split into five independent, non-array SLURM scripts. Every submission handles exactly one explicitly specified dataset/model/seed combination. No script expands an experimental grid internally.

| Stage | Script | One submission handles |
|---|---|---|
| 1 | `01_generate_data.sh` | One dataset |
| 2 | `02_train_choice_models.sh` | One GAsN or RAsN checkpoint |
| 3 | `03_train_diffusion.sh` | One diffusion checkpoint |
| 4 | `04_run_mip.sh` | One MIP solve |
| 5 | `05_run_d3ao.sh` | One D3AO sampling run |

Specify a single case through exported variables. For example, generate the MMNL, \(N=100\), \(\beta=1\), dataset-seed-0 data:

```bash
sbatch --export=ALL,CHOICE_MODEL=mmnl,N_PRODUCT=100,BETA=1.0,DATA_SEED=0 \
  rebuttal/meta_review/scripts/01_generate_data.sh
```

Train one GAsN checkpoint for that dataset:

```bash
sbatch --export=ALL,CHOICE_MODEL=mmnl,N_PRODUCT=100,BETA=1.0,DATA_SEED=0,REWARD_MODEL=gasn \
  rebuttal/meta_review/scripts/02_train_choice_models.sh
```

Run one GAsN MIP after that checkpoint exists:

```bash
sbatch --export=ALL,CHOICE_MODEL=mmnl,N_PRODUCT=100,BETA=1.0,DATA_SEED=0,REWARD_MODEL=gasn \
  rebuttal/meta_review/scripts/04_run_mip.sh
```

Run one GAsN-guided D3AO sampling replicate:

```bash
sbatch --export=ALL,CHOICE_MODEL=mmnl,N_PRODUCT=100,BETA=1.0,DATA_SEED=0,REWARD_MODEL=gasn,SAMPLING_SEED=0 \
  rebuttal/meta_review/scripts/05_run_d3ao.sh
```

Every script requests the `biostat` partition/account, 20 CPU cores, 200 GB memory, and a 20-hour wall-clock limit. SLURM logs are written under `/hpc/group/xielab/xj58/sbatch_output/`. The shared helper initializes Conda and activates `SpaRest` before running commands.

Formal experiment artifacts are stored outside the repository under:

```text
/hpc/home/xj58/da_rebuttal/
├── datasets/   # Generated logged assortment-choice JSON datasets
└── outputs/    # Checkpoints, manifests, logs, MIP results, and D3AO samples
```

Each stage creates both directories automatically. Override the common root with `ARTIFACT_ROOT`, or override `DATA_ROOT` and `OUTPUT_ROOT` separately when needed.

Its primary comparison is GAsN-reward MIP versus GAsN-guided D3AO, and RAsN-reward MIP versus RAsN-guided D3AO. Each D3AO run generates 256 candidates and returns the candidate selected by fitted reward. Ground-truth reward is used only for post-selection evaluation.

Useful inputs are `CHOICE_MODEL`, `N_PRODUCT`, `BETA`, `DATA_SEED`, `REWARD_MODEL`, and `SAMPLING_SEED`. Additional overrides are `PYTHON_BIN`, `CONDA_ENV`, `N_SAMPLES`, `DEVICE`, `NUM_CANDIDATES`, `GUIDANCE_SCALE`, `GUIDANCE_GAMMA`, `FORCE`, `ARTIFACT_ROOT`, `DATA_ROOT`, and `OUTPUT_ROOT`.

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
