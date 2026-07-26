#!/bin/bash

# Shared configuration and case mapping for the formal SLURM stages.
# This file is sourced by the numbered stage scripts and is not submitted itself.

initialize_formal_environment() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
  REPO_ROOT="$(cd "${script_dir}/../../.." && pwd)"
  cd "${REPO_ROOT}"

  CONDA_ENV="${CONDA_ENV:-SpaRest}"
  if [[ -n "${CONDA_EXE:-}" && -f "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh" ]]; then
    source "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  else
    echo "Could not locate conda.sh. Export CONDA_EXE or update formal_common.sh." >&2
    exit 1
  fi
  conda activate "${CONDA_ENV}"

  PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
  CHOICE_MODEL="${CHOICE_MODEL:-mmnl}"
  N_PRODUCT="${N_PRODUCT:-100}"
  BETA="${BETA:-1.0}"
  DATA_SEED="${DATA_SEED:-0}"
  REWARD_MODEL="${REWARD_MODEL:-gasn}"
  SAMPLING_SEED="${SAMPLING_SEED:-0}"
  N_SAMPLES="${N_SAMPLES:-10000}"
  DEVICE="${DEVICE:-cpu}"
  NUM_CANDIDATES="${NUM_CANDIDATES:-256}"
  GUIDANCE_SCALE="${GUIDANCE_SCALE:-1000}"
  GUIDANCE_GAMMA="${GUIDANCE_GAMMA:-3}"
  FORCE="${FORCE:-0}"
  ARTIFACT_ROOT="${ARTIFACT_ROOT:-/hpc/home/xj58/da_rebuttal}"
  DATA_ROOT="${DATA_ROOT:-${ARTIFACT_ROOT}/datasets}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-${ARTIFACT_ROOT}/outputs}"
  RUNNER="${REPO_ROOT}/rebuttal/meta_review/scripts/run_neural_mip_suite.py"
  mkdir -p "${DATA_ROOT}" "${OUTPUT_ROOT}"

}

select_data_case() {
  CASE_CHOICE_MODEL="${CHOICE_MODEL}"
  CASE_N_PRODUCTS="${N_PRODUCT}"
  CASE_BETA="${BETA}"
  CASE_DATA_SEED="${DATA_SEED}"
  CASE_BETA_TAG="$("${PYTHON_BIN}" -c 'import sys; print(format(float(sys.argv[1]), "g").replace("-", "m").replace(".", "p"))' "${CASE_BETA}")"
  CASE_NAME="${CASE_CHOICE_MODEL}_n${CASE_N_PRODUCTS}_beta${CASE_BETA_TAG}_seed${CASE_DATA_SEED}"
  CASE_DATASET="${DATA_ROOT}/${CASE_NAME}.json"
  CASE_OUTPUT="${OUTPUT_ROOT}/${CASE_NAME}"
}

print_stage_context() {
  echo "Stage: $1"
  echo "Case: ${CASE_NAME}"
  echo "Conda environment: ${CONDA_ENV}"
  echo "Python: ${PYTHON_BIN}"
  echo "Dataset: ${CASE_DATASET}"
  echo "Output: ${CASE_OUTPUT}"
}

force_argument() {
  if [[ "${FORCE}" == "1" ]]; then
    printf '%s' '--force'
  fi
}
