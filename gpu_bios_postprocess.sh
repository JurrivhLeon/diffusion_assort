#!/bin/bash
#SBATCH -p biostat-gpu
#SBATCH -A biostat
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=100G
#SBATCH -t 24:00:00
#SBATCH -J bios_post
#SBATCH --output=/hpc/group/xielab/xj58/sbatch_output/%x_output_%j.txt
#SBATCH --error=/hpc/group/xielab/xj58/sbatch_output/%x_error_%j.txt
#SBATCH --mail-user=xj58@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/.bashrc
conda activate SpaRest

set -euo pipefail

SCRIPT_DIR="/hpc/home/xj58/diffusion_assort"
DATA_ROOT="/hpc/home/xj58/diffusion_assort_datasets/generated_mnl"
CHOICE_ROOT="/hpc/home/xj58/diffusion_assort_outputs/choice_models"
DIFFUSION_ROOT="/hpc/home/xj58/diffusion_assort_outputs/diffusion_models"
EVAL_ROOT="/hpc/home/xj58/diffusion_assort_outputs/eval_mnl"

mkdir -p "${EVAL_ROOT}"

for N in 20 40 60 80 100; do
  for BETA in 0.1 1; do
    BETA_TAG="${BETA//./p}"
    for REPLICA in $(seq 1 10); do
      RUN_TAG="N${N}_beta${BETA_TAG}_dataset${REPLICA}"
      DATA_PATH="${DATA_ROOT}/N${N}_beta${BETA_TAG}/dataset_${REPLICA}.json"
      CHOICE_RUN_ROOT="${CHOICE_ROOT}/${RUN_TAG}"
      DIFFUSION_RUN_ROOT="${DIFFUSION_ROOT}/${RUN_TAG}"
      EVAL_RUN_ROOT="${EVAL_ROOT}/${RUN_TAG}"

      OPTIMAL_PATH="${EVAL_RUN_ROOT}/optimal_true_mnl.json"
      GUIDED_GASN_PATH="${EVAL_RUN_ROOT}/guided_gasn_scale1000_gamma3_n256.json"
      GUIDED_RASN_PATH="${EVAL_RUN_ROOT}/guided_rasn_scale1000_gamma3_n256.json"
      REPORT_PATH="${EVAL_RUN_ROOT}/report.json"

      mkdir -p "${EVAL_RUN_ROOT}"

      if [ ! -f "${OPTIMAL_PATH}" ]; then
        python3 "${SCRIPT_DIR}/true_mnl_optimal.py" \
          --data "${DATA_PATH}" \
          --output "${OPTIMAL_PATH}"
      fi

      if [ ! -f "${GUIDED_GASN_PATH}" ]; then
        python3 "${SCRIPT_DIR}/sample_guided_diffusion.py" \
          --data "${DATA_PATH}" \
          --checkpoint "${DIFFUSION_RUN_ROOT}/best.pt" \
          --choice-checkpoint "${CHOICE_RUN_ROOT}/gasn/best.pt" \
          --policy model \
          --num-samples 256 \
          --guidance-scale 1000 \
          --guidance-gamma 3 \
          --device cuda \
          --seed 123 \
          --output "${GUIDED_GASN_PATH}"
      fi

      if [ ! -f "${GUIDED_RASN_PATH}" ]; then
        python3 "${SCRIPT_DIR}/sample_guided_diffusion.py" \
          --data "${DATA_PATH}" \
          --checkpoint "${DIFFUSION_RUN_ROOT}/best.pt" \
          --choice-checkpoint "${CHOICE_RUN_ROOT}/rasn/best.pt" \
          --policy model \
          --num-samples 256 \
          --guidance-scale 1000 \
          --guidance-gamma 3 \
          --device cuda \
          --seed 123 \
          --output "${GUIDED_RASN_PATH}"
      fi

      python3 "${SCRIPT_DIR}/summarize_mnl_experiment.py" \
        --data "${DATA_PATH}" \
        --choice-root "${CHOICE_RUN_ROOT}" \
        --diffusion-root "${DIFFUSION_RUN_ROOT}" \
        --optimal "${OPTIMAL_PATH}" \
        --guided-gasn "${GUIDED_GASN_PATH}" \
        --guided-rasn "${GUIDED_RASN_PATH}" \
        --output "${REPORT_PATH}"
    done
  done
done

python3 "${SCRIPT_DIR}/summarize_mnl_suite.py" \
  --root "${EVAL_ROOT}" \
  --output "${EVAL_ROOT}/suite_summary.json"
