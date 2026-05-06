# diffusion_assort

Research code for assortment data generation, choice-model training, diffusion-based sampling, and optimization/evaluation.

## Environment

- Python 3.10+
- PyTorch
- NumPy
- Optional: Gurobi (required for `mip_optimize_assortment.py`)

## Script Reference

### Data and Dataset Utilities

- `generate_assortment_data.py`
  - Generates synthetic assortment-choice datasets.
  - Supports different choice models and assortment generation policies.
  - Writes dataset JSON used by training/evaluation scripts.

- `assortment_dataset.py`
  - Dataset loading utilities for assortment-choice JSON data.
  - Defines dataset structures used by choice-model training/evaluation.

- `diffusion_dataset.py`
  - Dataset utilities for diffusion training/sampling pipelines.

- `mccm_generation.py`
  - Helpers for generating or configuring MCCM-related synthetic choice settings.

### Choice Models and Training

- `choice_models.py`
  - Core neural/nonlinear choice model implementations (e.g., GAsN/RAsN variants).
  - Shared model definitions used by training and guided sampling.

- `train_choice_model.py`
  - Trains choice models from generated datasets.
  - Saves checkpoints for later evaluation or guided diffusion.

- `markov_chain_choice.py`
  - Markov-chain-based choice process/model utilities used by generation/evaluation components.

### Diffusion Models and Sampling

- `bernoulli_diffusion.py`
  - Bernoulli diffusion model components and sampling/training primitives.

- `train_bernoulli_diffusion.py`
  - Trains diffusion models on assortment datasets.
  - Saves checkpoints for unguided/guided sampling.

- `sample_unguided_diffusion.py`
  - Draws assortments from a trained diffusion checkpoint without guidance.

- `sample_guided_diffusion.py`
  - Draws assortments using diffusion guided by a trained choice model.

### Optimization and Evaluation

- `evaluate_assortment.py`
  - Evaluates generated/candidate assortments under configured reward/objective functions.

- `exhaustive_search_assortment.py`
  - Brute-force search baseline for optimal assortments on small problem sizes.

- `mnl_optimize_assortment.py`
  - Computes optimal assortments under MNL assumptions.

- `mmnl_optimize_assortment.py`
  - Computes optimal assortments under MMNL assumptions.

- `mccm_optimize_assortment.py`
  - Optimization routines for MCCM-based objectives/settings.

- `mccm_em_optimize_assortment.py`
  - MCCM optimization using EM-style fitting/estimation components.

- `mip_optimize_assortment.py`
  - Mixed-integer programming formulation/solver path for assortment optimization.
  - Typically requires Gurobi.

- `true_mnl_optimal.py`
  - Utility script to compute/report true MNL optimum for a dataset/setting.

### Analysis and Summaries

- `compare_boltzmann_optimization.py`
  - Compares optimization outcomes across Boltzmann-related generation/selection settings.

- `summarize_mnl_experiment.py`
  - Aggregates and summarizes outputs from MNL-focused experiment runs.

- `summarize_mnl_suite.py`
  - Produces summary tables/metrics across multiple MNL experiment configurations.

## Notes

- Generated datasets, model checkpoints, and large outputs are not tracked by default.
- Use per-script `--help` to see exact CLI options and defaults.
