#!/bin/bash
#SBATCH -p biostat-gpu
#SBATCH -A biostat
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -J dist_shift_n40n80
#SBATCH --output=/hpc/group/xielab/xj58/sbatch_output/%x_output_%j.txt
#SBATCH --error=/hpc/group/xielab/xj58/sbatch_output/%x_error_%j.txt
#SBATCH --mail-user=xj58@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ---------------------------
# Env
# ---------------------------
source ~/.bashrc
conda activate SpaRest

# ---------------------------
# Config
# ---------------------------
ROOT_DIR="/hpc/home/xj58/diffusion_assort"
cd "${ROOT_DIR}"

OUT_DATA_DIR="/hpc/home/xj58/datasets/distribution_shift"
OUT_DIFF_DIR="/hpc/home/xj58/outputs/distribution_shift/diffusion"

N_SAMPLES=10000
SEED=20260412

# Required by your request:
MCMC_BURN_IN=50
MCMC_THINNING=1
DIFFUSION_STEPS=100
TRAIN_SEED=42

# Toggle phases if needed.
RUN_DATA_GENERATION=1
RUN_DIFFUSION_TRAIN=1

mkdir -p "${OUT_DATA_DIR}" "${OUT_DIFF_DIR}"

echo "[INFO] Start: $(date)"
echo "[INFO] ROOT_DIR=${ROOT_DIR}"
echo "[INFO] N_SAMPLES=${N_SAMPLES}, SEED=${SEED}"
echo "[INFO] MCMC_BURN_IN=${MCMC_BURN_IN}, MCMC_THINNING=${MCMC_THINNING}, DIFFUSION_STEPS=${DIFFUSION_STEPS}"

if [[ "${RUN_DATA_GENERATION}" -eq 1 ]]; then
  echo "[PHASE 1] Generate non-Boltzmann datasets (uniform size -> uniform subset)"

  python -u generate_assortment_data.py \
    --output "${OUT_DATA_DIR}/nonboltzmann_mnl_n40_n10000.json" \
    --n-products 40 \
    --n-samples "${N_SAMPLES}" \
    --choice-model mnl \
    --assortment-policy uniform_size_then_subset \
    --seed "${SEED}" \
    --mcmc-burn-in "${MCMC_BURN_IN}" \
    --mcmc-thinning "${MCMC_THINNING}"

  python -u generate_assortment_data.py \
    --output "${OUT_DATA_DIR}/nonboltzmann_mmnl_n40_n10000.json" \
    --n-products 40 \
    --n-samples "${N_SAMPLES}" \
    --choice-model mmnl \
    --assortment-policy uniform_size_then_subset \
    --seed "${SEED}" \
    --mcmc-burn-in "${MCMC_BURN_IN}" \
    --mcmc-thinning "${MCMC_THINNING}"

  python -u generate_assortment_data.py \
    --output "${OUT_DATA_DIR}/nonboltzmann_mnl_n80_n10000.json" \
    --n-products 80 \
    --n-samples "${N_SAMPLES}" \
    --choice-model mnl \
    --assortment-policy uniform_size_then_subset \
    --seed "${SEED}" \
    --mcmc-burn-in "${MCMC_BURN_IN}" \
    --mcmc-thinning "${MCMC_THINNING}"

  python -u generate_assortment_data.py \
    --output "${OUT_DATA_DIR}/nonboltzmann_mmnl_n80_n10000.json" \
    --n-products 80 \
    --n-samples "${N_SAMPLES}" \
    --choice-model mmnl \
    --assortment-policy uniform_size_then_subset \
    --seed "${SEED}" \
    --mcmc-burn-in "${MCMC_BURN_IN}" \
    --mcmc-thinning "${MCMC_THINNING}"
fi

if [[ "${RUN_DIFFUSION_TRAIN}" -eq 1 ]]; then
  echo "[PHASE 2] Train Bernoulli diffusion with num_steps=${DIFFUSION_STEPS}"

  for dataset in \
    nonboltzmann_mnl_n40_n10000 \
    nonboltzmann_mmnl_n40_n10000 \
    nonboltzmann_mnl_n80_n10000 \
    nonboltzmann_mmnl_n80_n10000
  do
    python -u train_bernoulli_diffusion.py \
      --data "${OUT_DATA_DIR}/${dataset}.json" \
      --num-steps "${DIFFUSION_STEPS}" \
      --beta-start 1e-4 \
      --beta-end 0.1 \
      --hidden-sizes 128,128 \
      --time-emb-dim 32 \
      --epochs 50 \
      --batch-size 128 \
      --lr 1e-3 \
      --weight-decay 0.0 \
      --val-fraction 0.2 \
      --seed "${TRAIN_SEED}" \
      --device cuda:0 \
      --patience 10 \
      --outdir "${OUT_DIFF_DIR}/${dataset}"
  done
fi

echo "[INFO] Done: $(date)"
