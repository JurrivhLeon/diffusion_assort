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
OUT_CHOICE_DIR="/hpc/home/xj58/outputs/distribution_shift/choice_models"
OUT_EVAL_DIR="/hpc/home/xj58/outputs/distribution_shift/eval"

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
RUN_CHOICE_MODEL_TRAIN=1
RUN_OPTIMAL_RATIO_EVAL=1

EVAL_NUM_SAMPLES=256
EVAL_SEEDS=(42 84 126 168 210 252 294 336 378 420)
EVAL_RUN_UNGUIDED=1
EVAL_RUN_GUIDED=1

mkdir -p "${OUT_DATA_DIR}" "${OUT_DIFF_DIR}" "${OUT_CHOICE_DIR}" "${OUT_EVAL_DIR}"

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

if [[ "${RUN_CHOICE_MODEL_TRAIN}" -eq 1 ]]; then
  echo "[PHASE 2.5] Train choice models (GAsN/RAsN) for guided sampling"

  for dataset in \
    nonboltzmann_mnl_n40_n10000 \
    nonboltzmann_mmnl_n40_n10000 \
    nonboltzmann_mnl_n80_n10000 \
    nonboltzmann_mmnl_n80_n10000
  do
    data_path="${OUT_DATA_DIR}/${dataset}.json"

    python -u train_choice_model.py \
      --data "${data_path}" \
      --model gasn \
      --epochs 50 \
      --batch-size 128 \
      --lr 1e-3 \
      --weight-decay 0.0 \
      --val-fraction 0.2 \
      --seed "${TRAIN_SEED}" \
      --device cuda:0 \
      --patience 10 \
      --outdir "${OUT_CHOICE_DIR}/${dataset}/gasn"

    python -u train_choice_model.py \
      --data "${data_path}" \
      --model rasn \
      --num-blocks 1 \
      --epochs 50 \
      --batch-size 128 \
      --lr 1e-3 \
      --weight-decay 0.0 \
      --val-fraction 0.2 \
      --seed "${TRAIN_SEED}" \
      --device cuda:0 \
      --patience 10 \
      --outdir "${OUT_CHOICE_DIR}/${dataset}/rasn"
  done
fi

if [[ "${RUN_OPTIMAL_RATIO_EVAL}" -eq 1 ]]; then
  echo "[PHASE 3] Evaluate optimal ratio (unguided + guided by GAsN/RAsN) over 10 seeds and export CSV"

  for dataset in \
    nonboltzmann_mnl_n40_n10000 \
    nonboltzmann_mmnl_n40_n10000 \
    nonboltzmann_mnl_n80_n10000 \
    nonboltzmann_mmnl_n80_n10000
  do
    data_path="${OUT_DATA_DIR}/${dataset}.json"
    ckpt_path="${OUT_DIFF_DIR}/${dataset}/best.pt"
    setting_dir="${OUT_EVAL_DIR}/${dataset}"
    mkdir -p "${setting_dir}"

    echo "[EVAL] Dataset=${dataset}"

    # 1) True optimal value under the ground-truth model in dataset metadata.
    if [[ "${dataset}" == *"_mmnl_"* ]]; then
      python -u mmnl_optimize_assortment.py \
        --data "${data_path}" \
        --mip-gap 0.0 \
        --threads 8 \
        --output "${setting_dir}/true_optimum.json"
    else
      python -u mnl_optimize_assortment.py \
        --data "${data_path}" \
        --output "${setting_dir}/true_optimum.json"
    fi

    # 2) 10-seed sampling/evaluation with trained diffusion checkpoint.
    if [[ "${EVAL_RUN_UNGUIDED}" -eq 1 ]]; then
      for eval_seed in "${EVAL_SEEDS[@]}"; do
        python -u sample_unguided_diffusion.py \
          --data "${data_path}" \
          --checkpoint "${ckpt_path}" \
          --num-samples "${EVAL_NUM_SAMPLES}" \
          --seed "${eval_seed}" \
          --device cuda:0 \
          --output "${setting_dir}/unguided_seed_${eval_seed}.json"
      done
    fi

    if [[ "${EVAL_RUN_GUIDED}" -eq 1 ]]; then
      for guide_model in gasn rasn; do
        choice_ckpt="${OUT_CHOICE_DIR}/${dataset}/${guide_model}/best.pt"
        for eval_seed in "${EVAL_SEEDS[@]}"; do
          python -u sample_guided_diffusion.py \
            --data "${data_path}" \
            --checkpoint "${ckpt_path}" \
            --choice-checkpoint "${choice_ckpt}" \
            --policy model \
            --num-samples "${EVAL_NUM_SAMPLES}" \
            --seed "${eval_seed}" \
            --device cuda:0 \
            --output "${setting_dir}/guided_${guide_model}_seed_${eval_seed}.json"
        done
      done
    fi

    # 3) Aggregate optimal ratio mean/std for this setting and each method.
    DATASET="${dataset}" SETTING_DIR="${setting_dir}" EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES}" python - <<'PY'
import glob
import json
import os
import statistics

dataset = os.environ["DATASET"]
setting_dir = os.environ["SETTING_DIR"]
eval_num_samples = int(os.environ["EVAL_NUM_SAMPLES"])

with open(os.path.join(setting_dir, "true_optimum.json"), "r", encoding="utf-8") as f:
    true_opt = float(json.load(f)["best_value"])
if true_opt <= 0.0:
    raise ValueError(f"true optimal value must be positive, got {true_opt}.")

def aggregate(method: str, pattern: str, value_key: str) -> None:
    seed_paths = sorted(glob.glob(os.path.join(setting_dir, pattern)))
    if not seed_paths:
        return

    rows = []
    ratios = []
    for path in seed_paths:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        seed = int(obj["seed"])
        method_value = float(obj[value_key])
        ratio = method_value / true_opt
        ratios.append(ratio)
        rows.append(
            {
                "seed": seed,
                "method_best_true_reward": method_value,
                "true_optimal_value": true_opt,
                "optimal_ratio": ratio,
                "result_path": path,
            }
        )

    summary = {
        "dataset": dataset,
        "method": method,
        "num_seeds": len(ratios),
        "num_samples_per_seed": eval_num_samples,
        "mean_optimal_ratio": float(statistics.fmean(ratios)),
        "std_optimal_ratio": float(statistics.stdev(ratios)) if len(ratios) >= 2 else 0.0,
        "min_optimal_ratio": float(min(ratios)),
        "max_optimal_ratio": float(max(ratios)),
        "rows": rows,
    }
    with open(
        os.path.join(setting_dir, f"optimal_ratio_summary_{method}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

aggregate("unguided", "unguided_seed_*.json", "best_true_reward")
aggregate("guided_gasn", "guided_gasn_seed_*.json", "best_guided_true_expected_reward")
aggregate("guided_rasn", "guided_rasn_seed_*.json", "best_guided_true_expected_reward")
PY
  done

  # 4) Merge per-setting summaries into one JSON and one CSV table.
  OUT_EVAL_DIR="${OUT_EVAL_DIR}" EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES}" python - <<'PY'
import csv
import glob
import json
import os

out_dir = os.environ["OUT_EVAL_DIR"]
eval_num_samples = int(os.environ["EVAL_NUM_SAMPLES"])
summary_paths = sorted(glob.glob(os.path.join(out_dir, "*", "optimal_ratio_summary_*.json")))
if not summary_paths:
    raise ValueError(f"No per-setting summaries found under {out_dir}.")

merged = []
for p in summary_paths:
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    name = obj["dataset"]
    # name format: nonboltzmann_{model}_n{N}_n10000
    parts = name.split("_")
    model = parts[1]
    N = int(parts[2].lstrip("n"))
    merged.append(
        {
            "dataset": name,
            "method": obj["method"],
            "model": model,
            "N": N,
            "num_seeds": int(obj["num_seeds"]),
            "num_samples_per_seed": eval_num_samples,
            "mean_optimal_ratio": float(obj["mean_optimal_ratio"]),
            "std_optimal_ratio": float(obj["std_optimal_ratio"]),
            "min_optimal_ratio": float(obj["min_optimal_ratio"]),
            "max_optimal_ratio": float(obj["max_optimal_ratio"]),
            "summary_path": p,
        }
    )

merged.sort(key=lambda x: (x["N"], x["model"], x["method"]))

json_path = os.path.join(out_dir, "optimal_ratio_summary_all.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2)
    f.write("\n")

csv_path = os.path.join(out_dir, "optimal_ratio_summary.csv")
fieldnames = [
    "dataset",
    "method",
    "model",
    "N",
    "num_seeds",
    "num_samples_per_seed",
    "mean_optimal_ratio",
    "std_optimal_ratio",
    "min_optimal_ratio",
    "max_optimal_ratio",
    "summary_path",
]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(merged)
PY
fi

echo "[INFO] Done: $(date)"
