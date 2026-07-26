#!/bin/bash
#SBATCH -p biostat
#SBATCH -A biostat
#SBATCH -c 20
#SBATCH --mem=200G
#SBATCH -t 20:00:00
#SBATCH -J da_mip
#SBATCH --output=/hpc/group/xielab/xj58/sbatch_output/%x_output_%A_%a.txt
#SBATCH --error=/hpc/group/xielab/xj58/sbatch_output/%x_error_%A_%a.txt
#SBATCH --mail-user=xj58@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
set -eo pipefail
SUBMIT_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${SUBMIT_ROOT}/rebuttal/meta_review/scripts/formal_common.sh"
initialize_formal_environment
set -u
select_data_case
reward_model="${REWARD_MODEL}"
print_stage_context "mip_${reward_model}"
"${PYTHON_BIN}" -c "import gurobipy as gp; print('Gurobi:', gp.gurobi.version())"
"${PYTHON_BIN}" "${RUNNER}" --python "${PYTHON_BIN}" --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" --choice-models "${CASE_CHOICE_MODEL}" \
  --reward-models "${reward_model}" --n-products "${CASE_N_PRODUCTS}" --betas "${CASE_BETA}" \
  --seeds "${CASE_DATA_SEED}" --n-samples "${N_SAMPLES}" --stages mip $(force_argument)
