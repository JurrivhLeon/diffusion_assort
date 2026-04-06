#!/bin/bash
#SBATCH -p biostat-gpu                     
#SBATCH -A biostat                           
#SBATCH --gres=gpu:1                        
#SBATCH -c 10                                
#SBATCH --mem=100G                        
#SBATCH -t 7:00:00                            
#SBATCH -J bios
#SBATCH --output=/hpc/group/xielab/xj58/sbatch_output/%x_output_%j.txt  
#SBATCH --error=/hpc/group/xielab/xj58/sbatch_output/%x_error_%j.txt  
#SBATCH --mail-user=xj58@duke.edu                 
#SBATCH --mail-type=BEGIN,END,FAIL                

# Load environment
source ~/.bashrc
conda activate SpaRest

set -euo pipefail

SCRIPT_DIR="/hpc/home/xj58/diffusion_assort"
OUTPUT_ROOT="/hpc/home/xj58/diffusion_assort_datasets/generated_mnl"
CHOICE_ROOT="/hpc/home/xj58/diffusion_assort_outputs/choice_models"
DIFFUSION_ROOT="/hpc/home/xj58/diffusion_assort_outputs/diffusion_models"

mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${CHOICE_ROOT}"
mkdir -p "${DIFFUSION_ROOT}"

for N in 20 40 60 80 100; do
  for BETA in 0.1 1; do
    BETA_TAG="${BETA//./p}"
    TARGET_DIR="${OUTPUT_ROOT}/N${N}_beta${BETA_TAG}"
    mkdir -p "${TARGET_DIR}"
    BETA_SEED_OFFSET=0
    if [ "${BETA}" = "1" ]; then
      BETA_SEED_OFFSET=1
    fi

    for REPLICA in $(seq 1 10); do
      SEED=$((100000 * N + 1000 * REPLICA + BETA_SEED_OFFSET))
      OUTPUT_PATH="${TARGET_DIR}/dataset_${REPLICA}.json"
      RUN_TAG="N${N}_beta${BETA_TAG}_dataset${REPLICA}"
      GASN_OUTDIR="${CHOICE_ROOT}/${RUN_TAG}/gasn"
      RASN_OUTDIR="${CHOICE_ROOT}/${RUN_TAG}/rasn"
      DIFFUSION_OUTDIR="${DIFFUSION_ROOT}/${RUN_TAG}"

      if [ ! -f "${OUTPUT_PATH}" ]; then
        python3 "${SCRIPT_DIR}/generate_assortment_data.py" \
          --output "${OUTPUT_PATH}" \
          --n-products "${N}" \
          --n-samples 10000 \
          --seed "${SEED}" \
          --choice-model mnl \
          --assortment-policy boltzmann \
          --boltzmann-beta "${BETA}" \
          --boltzmann-sampler mcmc \
          --mcmc-device cuda
      fi

      if [ ! -f "${GASN_OUTDIR}/best.pt" ]; then
        python3 "${SCRIPT_DIR}/train_choice_model.py" \
          --data "${OUTPUT_PATH}" \
          --model gasn \
          --device cuda \
          --seed "${SEED}" \
          --outdir "${GASN_OUTDIR}"
      fi

      if [ ! -f "${RASN_OUTDIR}/best.pt" ]; then
        python3 "${SCRIPT_DIR}/train_choice_model.py" \
          --data "${OUTPUT_PATH}" \
          --model rasn \
          --device cuda \
          --seed "${SEED}" \
          --outdir "${RASN_OUTDIR}"
      fi

      if [ ! -f "${DIFFUSION_OUTDIR}/best.pt" ]; then
        python3 "${SCRIPT_DIR}/train_bernoulli_diffusion.py" \
          --data "${OUTPUT_PATH}" \
          --device cuda \
          --seed "${SEED}" \
          --outdir "${DIFFUSION_OUTDIR}"
      fi
    done
  done
done
