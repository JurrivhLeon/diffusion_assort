#!/usr/bin/env python3
"""Generate paper-style data, fit GAsN/RAsN, and optimize both with MIP.

The runner uses one generated dataset for both neural choice models. This keeps
the logged data, reward vector, train/validation split seed, and downstream MIP
objective comparable across GAsN and RAsN.

Examples
--------
Print the commands for a small pilot::

    python rebuttal/meta_review/scripts/run_neural_mip_suite.py \
      --n-products 20 --choice-models mnl --betas 1.0 --seeds 0 \
      --stages generate train mip --dry-run

Run only generation and training::

    python rebuttal/meta_review/scripts/run_neural_mip_suite.py \
      --n-products 20 --choice-models mnl --betas 1.0 --seeds 0 \
      --stages generate train

Resume and run MIP after moving to an environment with Gurobi::

    python rebuttal/meta_review/scripts/run_neural_mip_suite.py \
      --n-products 20 --choice-models mnl --betas 1.0 --seeds 0 \
      --stages mip
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Iterable


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_CONFIG = SCRIPT_PATH.parents[1] / "configs" / "neural_mip_suite.json"
DEFAULT_DATA_ROOT = REPO_ROOT / "datasets" / "rebuttal_neural_mip"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs_rebuttal_neural_mip"


def _csv_values(raw: str | None, cast: type) -> list[Any] | None:
    if raw is None:
        return None
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated list.")
    return [cast(value) for value in values]


def _beta_tag(beta: float) -> str:
    return format(beta, "g").replace("-", "m").replace(".", "p")


def _case_name(choice_model: str, n_products: int, beta: float, seed: int) -> str:
    return f"{choice_model}_n{n_products}_beta{_beta_tag(beta)}_seed{seed}"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _command_text(command: Iterable[str]) -> str:
    return shlex.join([str(part) for part in command])


def _run_command(
    command: list[str],
    *,
    stage_dir: Path,
    dry_run: bool,
    expected_output: Path,
    force: bool,
    parse_json_stdout: bool = False,
) -> str:
    if expected_output.exists() and not force:
        print(f"SKIP existing: {expected_output.relative_to(REPO_ROOT)}")
        return "skipped"

    print(_command_text(command), flush=True)
    if dry_run:
        return "dry-run"

    stage_dir.mkdir(parents=True, exist_ok=True)
    _write_json(stage_dir / "command.json", {"argv": command, "cwd": str(REPO_ROOT)})
    started = time.time()
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (stage_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (stage_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    _write_json(
        stage_dir / "status.json",
        {
            "returncode": completed.returncode,
            "runtime_seconds": time.time() - started,
            "expected_output": str(expected_output),
        },
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}. "
            f"See {stage_dir / 'stderr.log'}"
        )

    if parse_json_stdout:
        json_start = completed.stdout.find("{")
        if json_start < 0:
            raise RuntimeError(f"MIP command produced no JSON object; see {stage_dir / 'stdout.log'}")
        try:
            result = json.loads(completed.stdout[json_start:])
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Could not parse MIP JSON; see {stage_dir / 'stdout.log'}") from exc
        _write_json(expected_output, result)

    if not expected_output.exists():
        raise RuntimeError(f"Command succeeded but expected output is missing: {expected_output}")
    return "completed"


def _check_gurobi(python: str) -> None:
    completed = subprocess.run(
        [python, "-c", "import gurobipy as gp; print(gp.gurobi.version())"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "The MIP stage requires gurobipy and a working Gurobi environment. "
            f"Interpreter: {python}. Error: {completed.stderr.strip()}"
        )


def _generate_command(
    python: str,
    config: dict[str, Any],
    dataset_path: Path,
    choice_model: str,
    n_products: int,
    beta: float,
    seed: int,
) -> list[str]:
    command = [
        python,
        str(REPO_ROOT / "generate_assortment_data.py"),
        "--output", str(dataset_path),
        "--n-products", str(n_products),
        "--n-samples", str(config["n_samples"]),
        "--seed", str(seed),
        "--choice-model", choice_model,
        "--assortment-policy", str(config["assortment_policy"]),
        "--boltzmann-beta", str(beta),
        "--boltzmann-sampler", str(config["boltzmann_sampler"]),
        "--mcmc-chains", str(config["mcmc_chains"]),
        "--mcmc-burn-in", str(config["mcmc_burn_in"]),
        "--mcmc-thinning", str(config["mcmc_thinning"]),
        "--mcmc-device", str(config["mcmc_device"]),
        "--reward-low", str(config["reward_low"]),
        "--reward-high", str(config["reward_high"]),
    ]
    return command


def _train_command(
    python: str,
    config: dict[str, Any],
    dataset_path: Path,
    model_name: str,
    n_products: int,
    seed: int,
    model_dir: Path,
) -> list[str]:
    training = config["choice_training"]
    command = [
        python,
        str(REPO_ROOT / "train_choice_model.py"),
        "--data", str(dataset_path),
        "--model", model_name,
        "--epochs", str(training["epochs"]),
        "--batch-size", str(training["batch_size"]),
        "--lr", str(training["learning_rate"]),
        "--weight-decay", str(training["weight_decay"]),
        "--val-fraction", str(training["validation_fraction"]),
        "--patience", str(training["patience"]),
        "--seed", str(seed),
        "--device", str(training["device"]),
        "--outdir", str(model_dir),
    ]
    if model_name == "gasn":
        hidden_sizes = training["gasn_hidden_sizes"]
        command.extend(["--hidden-sizes", str(n_products if hidden_sizes == "N" else hidden_sizes)])
    else:
        command.extend(["--num-blocks", str(training["rasn_num_blocks"])])
    return command


def _mip_command(
    python: str,
    config: dict[str, Any],
    dataset_path: Path,
    checkpoint_path: Path,
) -> list[str]:
    mip = config["mip"]
    command = [
        python,
        str(REPO_ROOT / "mip_optimize_assortment.py"),
        "--data", str(dataset_path),
        "--checkpoint", str(checkpoint_path),
        "--policy", "model",
        "--backend", "mip",
        "--exp-pieces", str(mip["exp_pieces"]),
        "--time-limit", str(mip["time_limit_seconds"]),
        "--max-iterations", str(mip["max_iterations"]),
        "--dinkelbach-tol", str(mip["dinkelbach_tolerance"]),
        "--json",
    ]
    if mip.get("solver_verbose"):
        command.append("--solver-verbose")
    if mip.get("max_cardinality") is not None:
        command.extend(["--max-cardinality", str(mip["max_cardinality"])])
    if mip.get("exact_cardinality") is not None:
        command.extend(["--exact-cardinality", str(mip["exact_cardinality"])])
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--python", default=sys.executable, help="Python interpreter used for child processes.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--choice-models", help="Comma-separated override, e.g. mnl,mmnl.")
    parser.add_argument(
        "--reward-models",
        help="Comma-separated fitted reward models to train/optimize: gasn,rasn.",
    )
    parser.add_argument("--n-products", help="Comma-separated override, e.g. 20,40,100.")
    parser.add_argument("--betas", help="Comma-separated override, e.g. 0.1,1.0.")
    parser.add_argument("--seeds", help="Comma-separated override, e.g. 0,1,2.")
    parser.add_argument("--n-samples", type=int, help="Override the configured offline sample size.")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("generate", "train", "mip"),
        default=("generate", "train", "mip"),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--force", action="store_true", help="Rerun stages whose expected output exists.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    choice_models = _csv_values(args.choice_models, str) or config["choice_models"]
    reward_models = _csv_values(args.reward_models, str) or ["gasn", "rasn"]
    n_products_values = _csv_values(args.n_products, int) or config["n_products"]
    betas = _csv_values(args.betas, float) or config["betas"]
    seeds = _csv_values(args.seeds, int) or config["seeds"]
    if args.n_samples is not None:
        if args.n_samples <= 0:
            raise ValueError("--n-samples must be positive.")
        config["n_samples"] = args.n_samples
    unknown_models = set(choice_models) - {"mnl", "mccm", "mmnl"}
    if unknown_models:
        raise ValueError(f"Unsupported data-generating choice models: {sorted(unknown_models)}")
    unknown_reward_models = set(reward_models) - {"gasn", "rasn"}
    if unknown_reward_models:
        raise ValueError(f"Unsupported fitted reward models: {sorted(unknown_reward_models)}")
    if any(value <= 0 for value in n_products_values):
        raise ValueError("All --n-products values must be positive.")
    if "mip" in args.stages and not args.dry_run:
        _check_gurobi(args.python)

    manifest: dict[str, Any] = {"config": config, "cases": []}
    for choice_model, n_products, beta, seed in product(
        choice_models, n_products_values, betas, seeds
    ):
        case = _case_name(choice_model, n_products, beta, seed)
        dataset_path = args.data_root / f"{case}.json"
        case_dir = args.output_root / case
        case_record: dict[str, Any] = {
            "case": case,
            "choice_model": choice_model,
            "n_products": n_products,
            "beta": beta,
            "seed": seed,
            "dataset": str(dataset_path),
            "stages": {},
        }
        if "generate" in args.stages:
            case_record["stages"]["generate"] = _run_command(
                _generate_command(args.python, config, dataset_path, choice_model, n_products, beta, seed),
                stage_dir=case_dir / "generate",
                dry_run=args.dry_run,
                expected_output=dataset_path,
                force=args.force,
            )
        elif not dataset_path.exists() and not args.dry_run:
            raise FileNotFoundError(f"Missing dataset required by later stages: {dataset_path}")

        for model_name in reward_models:
            model_dir = case_dir / model_name
            checkpoint_path = model_dir / "best.pt"
            if "train" in args.stages:
                case_record["stages"][f"train_{model_name}"] = _run_command(
                    _train_command(
                        args.python, config, dataset_path, model_name, n_products, seed, model_dir
                    ),
                    stage_dir=model_dir / "runner_logs",
                    dry_run=args.dry_run,
                    expected_output=checkpoint_path,
                    force=args.force,
                )
            elif not checkpoint_path.exists() and "mip" in args.stages and not args.dry_run:
                raise FileNotFoundError(f"Missing checkpoint required by MIP: {checkpoint_path}")

            if "mip" in args.stages:
                mip_dir = case_dir / f"mip_{model_name}"
                result_path = mip_dir / "result.json"
                case_record["stages"][f"mip_{model_name}"] = _run_command(
                    _mip_command(args.python, config, dataset_path, checkpoint_path),
                    stage_dir=mip_dir,
                    dry_run=args.dry_run,
                    expected_output=result_path,
                    force=args.force,
                    parse_json_stdout=True,
                )
        manifest["cases"].append(case_record)

    if not args.dry_run:
        if len(manifest["cases"]) == 1:
            stage_tag = "_".join(args.stages)
            manifest_path = (
                args.output_root
                / "manifests"
                / f"{manifest['cases'][0]['case']}_{stage_tag}.json"
            )
        else:
            manifest_path = args.output_root / "suite_manifest.json"
        _write_json(manifest_path, manifest)
    stage_summary = ", ".join(args.stages)
    print(
        f"Prepared {len(manifest['cases'])} dataset cases with reward models "
        f"{','.join(reward_models)}; "
        f"requested stages: {stage_summary}."
    )


if __name__ == "__main__":
    main()
