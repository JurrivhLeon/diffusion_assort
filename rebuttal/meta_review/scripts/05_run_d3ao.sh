#!/bin/bash
#SBATCH -p biostat
#SBATCH -A biostat
#SBATCH -c 20
#SBATCH --mem=200G
#SBATCH -t 20:00:00
#SBATCH -J da_d3ao
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
sampling_seed="${SAMPLING_SEED}"
print_stage_context "d3ao_${reward_model}_sampling_seed${sampling_seed}"
diffusion_checkpoint="${CASE_OUTPUT}/diffusion/best.pt"
choice_checkpoint="${CASE_OUTPUT}/${reward_model}/best.pt"
guided_dir="${CASE_OUTPUT}/d3ao_${reward_model}"
mkdir -p "${guided_dir}/logs"
output="${guided_dir}/seed${sampling_seed}.json"
log="${guided_dir}/logs/seed${sampling_seed}.log"
if [[ -f "${output}" && "${FORCE}" != "1" ]]; then echo "SKIP existing: ${output}"; exit 0; fi
"${PYTHON_BIN}" sample_guided_diffusion.py --data "${CASE_DATASET}" \
  --checkpoint "${diffusion_checkpoint}" --choice-checkpoint "${choice_checkpoint}" \
  --policy model --num-samples "${NUM_CANDIDATES}" --guidance-scale "${GUIDANCE_SCALE}" \
  --guidance-gamma "${GUIDANCE_GAMMA}" --seed "${sampling_seed}" \
  --device "${DEVICE}" --output "${output}" >"${log}" 2>&1
