#!/usr/bin/env python3
"""Repair MMNL outputs after fixing the exact reference solver.

This script recomputes the true MMNL optimal assortment for every case under
``outputs_mmnl/N*_beta*/seed*`` and then refreshes the dependent JSON artifacts:

- ``reference_mmnl_milp.json``
- ``mnl_mle_on_mmnl.json``
- guided-sampling ``optimal_ratio_stats`` inside the per-case ``guided_*.json``
- ``report.json``

It does not retrain models or resample diffusion outputs; it only rewrites the
stored summaries using the already-saved raw sample rewards.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from assortment_dataset import load_assortment_json
from mmnl_optimize_assortment import solve_mmnl_optimal_assortment


ROOT = Path("/hpc/home/jl1344/diffusion_assort")
OUT_ROOT = ROOT / "outputs_mmnl"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _sanitize(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _sanitize(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_sanitize(val) for val in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(payload), indent=2, sort_keys=True) + "\n")


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(values[lo])
    w = pos - lo
    return float(values[lo] * (1.0 - w) + values[hi] * w)


def _ratio_stats(true_rewards: list[float], ref_reward: float) -> dict[str, float]:
    if ref_reward == 0.0:
        ratios = [float("nan") for _ in true_rewards]
    else:
        ratios = [float(x) / ref_reward for x in true_rewards]
    finite = [x for x in ratios if math.isfinite(x)]
    if not finite:
        return {"max": float("nan"), "mean": float("nan"), "q50": float("nan"), "q90": float("nan")}
    return {
        "max": float(max(finite)),
        "mean": float(statistics.fmean(finite)),
        "q50": _quantile(finite, 0.5),
        "q90": _quantile(finite, 0.9),
    }


def _extract_case_dir(report_path: Path) -> Path:
    return report_path.parent


def _refresh_guided_file(path: Path, ref_reward: float) -> dict[str, float]:
    payload = _load_json(path)
    payload["optimal_ratio_stats"] = _ratio_stats(
        [float(x) for x in payload["true_expected_rewards"]], ref_reward
    )
    _write_json(path, payload)
    return payload["optimal_ratio_stats"]


def _guided_method_key(path: Path) -> str:
    name = path.name
    if name.startswith("guided_gasn_"):
        return "gasn"
    if name.startswith("guided_rasn_"):
        return "rasn"
    if name.startswith("guided_tcn"):
        return "tcn"
    raise ValueError(f"Unrecognized guided sampling file name: {name}")


def repair_case(report_path: Path) -> dict[str, Any]:
    case_dir = _extract_case_dir(report_path)
    dataset_meta = _load_json(case_dir / "dataset.json")
    data_path = Path(dataset_meta["path"])
    raw = load_assortment_json(data_path)
    if "meta" not in raw:
        raise ValueError(f"Dataset at {data_path} does not contain a meta field.")
    meta = raw["meta"]

    reference = solve_mmnl_optimal_assortment(meta, solver_verbose=False)
    reference["data"] = str(data_path)
    reference_path = case_dir / "reference_mmnl_milp.json"
    _write_json(reference_path, reference)

    ref_reward = float(reference["best_value"])

    mnl_mle_path = case_dir / "mnl_mle_on_mmnl.json"
    if mnl_mle_path.exists():
        mnl_mle = _load_json(mnl_mle_path)
        mnl_mle["reference_optimal_reward"] = ref_reward
        if mnl_mle.get("true_expected_reward") is not None:
            mnl_mle["optimal_ratio"] = float(mnl_mle["true_expected_reward"]) / ref_reward
        _write_json(mnl_mle_path, mnl_mle)
    else:
        mnl_mle = None

    guided_ratios: dict[str, dict[str, float]] = {}
    for guided_path in sorted(case_dir.glob("guided_*_scale1000_gamma3_n256.json")):
        method = _guided_method_key(guided_path)
        guided_ratios[method] = _refresh_guided_file(guided_path, ref_reward)

    report_path = case_dir / "report.json"
    report = _load_json(report_path) if report_path.exists() else {}
    report["reference"] = _load_json(reference_path)
    if mnl_mle is not None:
        report["mnl_mle"] = mnl_mle
    if "guided_sampling" in report:
        for method, stats in guided_ratios.items():
            if method in report["guided_sampling"]:
                report["guided_sampling"][method]["optimal_ratio_stats"] = stats
    _write_json(report_path, report)

    return {
        "case_dir": str(case_dir),
        "data": str(data_path),
        "reference_reward": ref_reward,
        "reference_assortment": reference["best_assortment_str"],
    }


def main() -> None:
    repaired = []
    for report_path in sorted(OUT_ROOT.glob("N*_beta*/seed*/report.json")):
        repaired.append(repair_case(report_path))
    print(json.dumps(_sanitize({"repaired_cases": len(repaired), "sample": repaired[:3]}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
