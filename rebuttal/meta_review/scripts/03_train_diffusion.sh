#!/bin/bash
#SBATCH -p biostat
#SBATCH -A biostat
#SBATCH -c 20
#SBATCH --mem=200G
#SBATCH -t 20:00:00
#SBATCH -J da_diff
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
print_stage_context "train_diffusion"
diffusion_dir="${CASE_OUTPUT}/diffusion"
checkpoint="${diffusion_dir}/best.pt"
if [[ -f "${checkpoint}" && "${FORCE}" != "1" ]]; then echo "SKIP existing: ${checkpoint}"; exit 0; fi
mkdir -p "${diffusion_dir}"
"${PYTHON_BIN}" train_bernoulli_diffusion.py --data "${CASE_DATASET}" \
  --hidden-sizes 128,128 --num-steps 100 --beta-start 0.0001 --beta-end 0.1 \
  --epochs 50 --batch-size 128 --lr 0.001 --weight-decay 0 --val-fraction 0.2 \
  --seed "${CASE_DATA_SEED}" --device "${DEVICE}" --patience 10 --outdir "${diffusion_dir}"
