#!/bin/bash
#SBATCH -p biostat
#SBATCH -A biostat
#SBATCH -c 20
#SBATCH --mem=200G
#SBATCH -t 20:00:00
#SBATCH -J mip_vs_d3ao
#SBATCH --output=/hpc/group/xielab/xj58/sbatch_output/%x_output_%j.txt
#SBATCH --error=/hpc/group/xielab/xj58/sbatch_output/%x_error_%j.txt
#SBATCH --mail-user=xj58@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=0-19%5

set -euo pipefail

# Formal rebuttal experiment: MIP versus D3AO under the same fitted rewards.
#
# Default scope:
#   - MMNL ground-truth data (the main misspecification setting)
#   - N=100
#   - behavior-policy beta in {0.1, 1.0}
#   - 10 independent dataset seeds
#   - GAsN and RAsN fitted rewards
#   - one MIP solve per fitted checkpoint
#   - 10 D3AO sampling seeds, 256 candidates per sampling run
#
# Override example:
#   sbatch --export=ALL,CHOICE_MODELS=mnl,N_PRODUCTS=20,BETAS=1.0,\
#DATA_SEEDS=0,SAMPLING_SEEDS=0,DEVICE=cpu,RUN_MIP=0 \
#     rebuttal/meta_review/scripts/run_formal_mip_vs_d3ao.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-SpaRest}"

# SLURM jobs use a non-interactive shell, so initialize Conda explicitly before
# activating the requested environment. CONDA_EXE is preferred when exported
# by the submission shell; the two common user installation paths are fallbacks.
if [[ -n "${CONDA_EXE:-}" && -f "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh" ]]; then
  source "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  echo "Could not locate conda.sh. Export CONDA_EXE or update the Conda initialization block." >&2
  exit 1
fi

conda activate "${CONDA_ENV}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
CHOICE_MODELS="${CHOICE_MODELS:-mmnl}"
N_PRODUCTS="${N_PRODUCTS:-100}"
BETAS="${BETAS:-0.1,1.0}"
DATA_SEEDS="${DATA_SEEDS:-0,1,2,3,4,5,6,7,8,9}"
SAMPLING_SEEDS="${SAMPLING_SEEDS:-0,1,2,3,4,5,6,7,8,9}"
N_SAMPLES="${N_SAMPLES:-10000}"
DEVICE="${DEVICE:-cuda}"
NUM_CANDIDATES="${NUM_CANDIDATES:-256}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1000}"
GUIDANCE_GAMMA="${GUIDANCE_GAMMA:-3}"
RUN_MIP="${RUN_MIP:-1}"
FORCE="${FORCE:-0}"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/hpc/home/xj58/da_rebuttal}"
DATA_ROOT="${DATA_ROOT:-${ARTIFACT_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ARTIFACT_ROOT}/outputs}"
RUNNER="${REPO_ROOT}/rebuttal/meta_review/scripts/run_neural_mip_suite.py"

mkdir -p "${DATA_ROOT}" "${OUTPUT_ROOT}"

if [[ "${FORCE}" == "1" ]]; then
  FORCE_ARG="--force"
else
  FORCE_ARG=""
fi

IFS=',' read -r -a CHOICE_MODEL_VALUES <<< "${CHOICE_MODELS}"
IFS=',' read -r -a N_PRODUCT_VALUES <<< "${N_PRODUCTS}"
IFS=',' read -r -a BETA_VALUES <<< "${BETAS}"
IFS=',' read -r -a DATA_SEED_VALUES <<< "${DATA_SEEDS}"
IFS=',' read -r -a SAMPLING_SEED_VALUES <<< "${SAMPLING_SEEDS}"

# Map one SLURM array task to one data-generating configuration. The default
# grid has 1 choice model x 1 N x 2 betas x 10 seeds = 20 tasks. When this
# script is run directly (without SLURM_ARRAY_TASK_ID), it retains the complete
# grid, which is useful for local pilots with explicit small overrides.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  CASE_GRID=()
  for grid_choice_model in "${CHOICE_MODEL_VALUES[@]}"; do
    for grid_n_products in "${N_PRODUCT_VALUES[@]}"; do
      for grid_beta in "${BETA_VALUES[@]}"; do
        for grid_data_seed in "${DATA_SEED_VALUES[@]}"; do
          CASE_GRID+=("${grid_choice_model}|${grid_n_products}|${grid_beta}|${grid_data_seed}")
        done
      done
    done
  done

  if (( SLURM_ARRAY_TASK_ID >= ${#CASE_GRID[@]} )); then
    echo "Array task ${SLURM_ARRAY_TASK_ID} has no case; configured grid has ${#CASE_GRID[@]} cases."
    exit 0
  fi

  IFS='|' read -r TASK_CHOICE_MODEL TASK_N_PRODUCTS TASK_BETA TASK_DATA_SEED \
    <<< "${CASE_GRID[SLURM_ARRAY_TASK_ID]}"
  CHOICE_MODELS="${TASK_CHOICE_MODEL}"
  N_PRODUCTS="${TASK_N_PRODUCTS}"
  BETAS="${TASK_BETA}"
  DATA_SEEDS="${TASK_DATA_SEED}"
  CHOICE_MODEL_VALUES=("${TASK_CHOICE_MODEL}")
  N_PRODUCT_VALUES=("${TASK_N_PRODUCTS}")
  BETA_VALUES=("${TASK_BETA}")
  DATA_SEED_VALUES=("${TASK_DATA_SEED}")
fi

echo "Repository: ${REPO_ROOT}"
echo "Conda environment: ${CONDA_ENV}"
echo "Python: ${PYTHON_BIN}"
echo "Choice models: ${CHOICE_MODELS}"
echo "N: ${N_PRODUCTS}"
echo "Betas: ${BETAS}"
echo "Dataset seeds: ${DATA_SEEDS}"
echo "D3AO sampling seeds: ${SAMPLING_SEEDS}"
echo "Device: ${DEVICE}"
echo "SLURM array task: ${SLURM_ARRAY_TASK_ID:-none}"
echo "Artifact root: ${ARTIFACT_ROOT}"
echo "Dataset root: ${DATA_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"

"${PYTHON_BIN}" -c "import torch; print('PyTorch:', torch.__version__)"
if [[ "${RUN_MIP}" == "1" ]]; then
  "${PYTHON_BIN}" -c "import gurobipy as gp; print('Gurobi:', gp.gurobi.version())"
fi

# Stage 1: generate each logged dataset and fit matched GAsN/RAsN models.
"${PYTHON_BIN}" "${RUNNER}" \
  --python "${PYTHON_BIN}" \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --choice-models "${CHOICE_MODELS}" \
  --n-products "${N_PRODUCTS}" \
  --betas "${BETAS}" \
  --seeds "${DATA_SEEDS}" \
  --n-samples "${N_SAMPLES}" \
  --stages generate train \
  ${FORCE_ARG}

# Stage 2: train one diffusion prior per dataset. Both fitted rewards use this
# exact same prior, so the GAsN/RAsN comparison changes only reward guidance.
for choice_model in "${CHOICE_MODEL_VALUES[@]}"; do
  for n_products in "${N_PRODUCT_VALUES[@]}"; do
    for beta in "${BETA_VALUES[@]}"; do
      beta_tag="$("${PYTHON_BIN}" -c 'import sys; print(format(float(sys.argv[1]), "g").replace("-", "m").replace(".", "p"))' "${beta}")"
      for data_seed in "${DATA_SEED_VALUES[@]}"; do
        case_name="${choice_model}_n${n_products}_beta${beta_tag}_seed${data_seed}"
        dataset_path="${DATA_ROOT}/${case_name}.json"
        case_dir="${OUTPUT_ROOT}/${case_name}"
        diffusion_dir="${case_dir}/diffusion"
        diffusion_checkpoint="${diffusion_dir}/best.pt"
        diffusion_log="${diffusion_dir}/formal_runner.log"

        if [[ -f "${diffusion_checkpoint}" && "${FORCE}" != "1" ]]; then
          echo "SKIP existing diffusion checkpoint: ${diffusion_checkpoint}"
        else
          mkdir -p "${diffusion_dir}"
          echo "Training diffusion: ${case_name}"
          "${PYTHON_BIN}" train_bernoulli_diffusion.py \
            --data "${dataset_path}" \
            --hidden-sizes 128,128 \
            --num-steps 100 \
            --beta-start 0.0001 \
            --beta-end 0.1 \
            --epochs 50 \
            --batch-size 128 \
            --lr 0.001 \
            --weight-decay 0 \
            --val-fraction 0.2 \
            --seed "${data_seed}" \
            --device "${DEVICE}" \
            --patience 10 \
            --outdir "${diffusion_dir}" \
            >"${diffusion_log}" 2>&1
        fi
      done
    done
  done
done

# Stage 3: solve each frozen GAsN/RAsN reward with MIP once. The Python runner
# performs a Gurobi preflight and records solver stdout, stderr, status, runtime,
# and the parsed JSON result for each checkpoint.
if [[ "${RUN_MIP}" == "1" ]]; then
  "${PYTHON_BIN}" "${RUNNER}" \
    --python "${PYTHON_BIN}" \
    --data-root "${DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --choice-models "${CHOICE_MODELS}" \
    --n-products "${N_PRODUCTS}" \
    --betas "${BETAS}" \
    --seeds "${DATA_SEEDS}" \
    --n-samples "${N_SAMPLES}" \
    --stages mip \
    ${FORCE_ARG}
else
  echo "Skipping MIP because RUN_MIP=${RUN_MIP}."
fi

# Stage 4: guided D3AO sampling. sample_guided_diffusion.py selects the returned
# candidate by the fitted GAsN/RAsN reward; true reward is evaluated only after
# selection and is stored as best_guided_true_expected_reward.
for choice_model in "${CHOICE_MODEL_VALUES[@]}"; do
  for n_products in "${N_PRODUCT_VALUES[@]}"; do
    for beta in "${BETA_VALUES[@]}"; do
      beta_tag="$("${PYTHON_BIN}" -c 'import sys; print(format(float(sys.argv[1]), "g").replace("-", "m").replace(".", "p"))' "${beta}")"
      for data_seed in "${DATA_SEED_VALUES[@]}"; do
        case_name="${choice_model}_n${n_products}_beta${beta_tag}_seed${data_seed}"
        dataset_path="${DATA_ROOT}/${case_name}.json"
        case_dir="${OUTPUT_ROOT}/${case_name}"
        diffusion_checkpoint="${case_dir}/diffusion/best.pt"

        for reward_model in gasn rasn; do
          choice_checkpoint="${case_dir}/${reward_model}/best.pt"
          guided_dir="${case_dir}/d3ao_${reward_model}"
          mkdir -p "${guided_dir}"

          for sampling_seed in "${SAMPLING_SEED_VALUES[@]}"; do
            guided_output="${guided_dir}/seed${sampling_seed}.json"
            guided_log="${guided_dir}/seed${sampling_seed}.log"
            if [[ -f "${guided_output}" && "${FORCE}" != "1" ]]; then
              echo "SKIP existing D3AO result: ${guided_output}"
              continue
            fi

            echo "D3AO: ${case_name}, reward=${reward_model}, sampling_seed=${sampling_seed}"
            "${PYTHON_BIN}" sample_guided_diffusion.py \
              --data "${dataset_path}" \
              --checkpoint "${diffusion_checkpoint}" \
              --choice-checkpoint "${choice_checkpoint}" \
              --policy model \
              --num-samples "${NUM_CANDIDATES}" \
              --guidance-scale "${GUIDANCE_SCALE}" \
              --guidance-gamma "${GUIDANCE_GAMMA}" \
              --seed "${sampling_seed}" \
              --device "${DEVICE}" \
              --output "${guided_output}" \
              >"${guided_log}" 2>&1
          done
        done
      done
    done
  done
done

echo "Formal MIP-versus-D3AO experiment completed."
echo "Outputs: ${OUTPUT_ROOT}"
