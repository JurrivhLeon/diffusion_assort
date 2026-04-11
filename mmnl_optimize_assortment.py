#!/usr/bin/env python3
"""Exact optimal assortment under the mixed multinomial logit model.

We solve the finite-mixture MNL assortment problem exactly with a MILP.

For each customer type c, let

    a_c = exp(u_{c,0})
    b_{c,i} = exp(u_{c,i})

and let x_i in {0,1} indicate whether product i is offered. The expected
reward contribution of type c is

    (r_0 a_c + sum_i r_i b_{c,i} x_i) / (a_c + sum_i b_{c,i} x_i)

This is a linear-fractional term. Using a Charnes-Cooper transform per type,
we introduce

    t_c = 1 / (a_c + sum_i b_{c,i} x_i)
    y_{c,i} = t_c x_i

which yields the linear constraints

    a_c t_c + sum_i b_{c,i} y_{c,i} = 1

and the exact linking constraints

    y_{c,i} = t_c x_i,  x_i in {0,1}, 0 <= t_c <= 1 / a_c.

The resulting problem is a MILP and gives an exact optimum for the MMNL model.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import gurobipy as gp
from gurobipy import GRB

from assortment_dataset import load_assortment_json


def assortment_to_string(assortment: Sequence[int]) -> str:
    return "".join(str(int(x)) for x in assortment)


def _load_mmnl_tensors(
    meta: dict[str, Any],
) -> tuple[list[list[float]], list[float], list[float], float, float]:
    utilities_by_type = meta.get("mmnl_utilities")
    type_probs = meta.get("mmnl_type_probs")
    rewards = meta.get("rewards")
    if utilities_by_type is None:
        raise ValueError("Metadata does not contain 'mmnl_utilities'.")
    if type_probs is None:
        raise ValueError("Metadata does not contain 'mmnl_type_probs'.")
    if rewards is None:
        raise ValueError("Metadata does not contain 'rewards'.")

    utilities_by_type = [[float(v) for v in row] for row in utilities_by_type]
    type_probs = [float(v) for v in type_probs]
    rewards = [float(v) for v in rewards]
    if not utilities_by_type:
        raise ValueError("mmnl_utilities must be non-empty.")
    if len(utilities_by_type) != len(type_probs):
        raise ValueError("mmnl_utilities and mmnl_type_probs must have the same length.")
    n_products = len(rewards)
    if any(len(row) != n_products for row in utilities_by_type):
        raise ValueError("Each MMNL utility row must have length N.")

    outside_utility = float(meta.get("mmnl_outside_utility", 0.0))
    reward_no_purchase = float(meta.get("reward_no_purchase", 0.0))
    return utilities_by_type, type_probs, rewards, outside_utility, reward_no_purchase


def solve_mmnl_optimal_assortment(
    meta: dict[str, Any],
    *,
    time_limit_sec: float | None = None,
    mip_gap: float = 0.0,
    solver_verbose: bool = False,
    threads: int | None = None,
) -> dict[str, Any]:
    """Return the exact optimal assortment under the MMNL model."""
    if meta.get("choice_model") != "mmnl":
        raise ValueError("This solver is only defined for datasets with choice_model='mmnl'.")

    utilities_by_type, type_probs, rewards, outside_utility, reward_no_purchase = _load_mmnl_tensors(meta)
    n_types = len(type_probs)
    n_products = len(rewards)

    if mip_gap < 0.0:
        raise ValueError("--mip-gap must be non-negative.")
    if time_limit_sec is not None and time_limit_sec <= 0.0:
        raise ValueError("--time-limit-sec must be positive if provided.")
    if threads is not None and threads <= 0:
        raise ValueError("--threads must be positive if provided.")

    # Normalize each type's logits before exponentiating to improve numerical
    # conditioning. The MMNL choice probabilities are invariant to an additive
    # shift of all logits within a type, and the fractional objective is also
    # invariant to scaling all weights for a type by a positive constant.
    #
    # For each type c we store weights in "probability simplex" form:
    #   a_c = exp(u_0 - s_c) / Z_c
    #   b_{c,i} = exp(u_i - s_c) / Z_c
    # where s_c is the max logit for the type and Z_c is the sum of shifted
    # weights. This keeps all coefficients in [0, 1] and sums them to 1.
    outside_weights: list[float] = []
    product_weights: list[list[float]] = []
    t_upper: list[float] = []
    for row in utilities_by_type:
        logits = [outside_utility] + [float(u) for u in row]
        shift = max(logits)
        shifted = [math.exp(logit - shift) for logit in logits]
        weight_sum = float(sum(shifted))
        if weight_sum <= 0.0:
            raise ValueError("MMNL weight normalization failed due to non-positive sum.")
        outside_w = shifted[0] / weight_sum
        prod_w = [w / weight_sum for w in shifted[1:]]
        outside_weights.append(outside_w)
        product_weights.append(prod_w)
        t_upper.append(1.0 / outside_w)

    model = gp.Model("mmnl_assortment")
    model.Params.OutputFlag = 1 if solver_verbose else 0
    model.Params.MIPGap = float(mip_gap)
    model.Params.FeasibilityTol = 1e-9
    model.Params.OptimalityTol = 1e-9
    model.Params.IntFeasTol = 1e-9
    model.Params.NumericFocus = 3
    model.Params.ScaleFlag = 2
    if time_limit_sec is not None:
        model.Params.TimeLimit = float(time_limit_sec)
    if threads is not None:
        model.Params.Threads = int(threads)

    x = model.addVars(n_products, vtype=GRB.BINARY, name="x")
    t = model.addVars(n_types, lb=0.0, vtype=GRB.CONTINUOUS, name="t")
    y = model.addVars(
        n_types,
        n_products,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="y",
    )

    for c in range(n_types):
        t[c].UB = t_upper[c]
        model.addConstr(
            outside_weights[c] * t[c]
            + gp.quicksum(product_weights[c][i] * y[c, i] for i in range(n_products))
            == 1.0,
            name=f"cc_normalize[{c}]",
        )
        for i in range(n_products):
            y[c, i].UB = t_upper[c]
            model.addConstr(y[c, i] <= t[c], name=f"link_ub_t[{c},{i}]")
            model.addConstr(y[c, i] <= t_upper[c] * x[i], name=f"link_ub_x[{c},{i}]")
            model.addConstr(
                y[c, i] >= t[c] - t_upper[c] * (1.0 - x[i]),
                name=f"link_lb[{c},{i}]",
            )

    # The MMNL benchmark datasets assign strictly positive rewards to every
    # product, so the empty assortment can never be optimal. Enforce at least
    # one offered product to avoid a degenerate zero-revenue solution.
    model.addConstr(gp.quicksum(x[i] for i in range(n_products)) >= 1, name="nonempty_assortment")

    model.setObjective(
        gp.quicksum(
            type_probs[c]
            * (
                reward_no_purchase * outside_weights[c] * t[c]
                + gp.quicksum(
                    rewards[i] * product_weights[c][i] * y[c, i]
                    for i in range(n_products)
                )
            )
            for c in range(n_types)
        ),
        GRB.MAXIMIZE,
    )

    model.optimize()

    if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        raise RuntimeError(f"Gurobi ended with unexpected status {model.Status}.")
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(
            "MMNL MILP did not reach optimality. "
            "Try increasing the time limit or relaxing the model if needed."
        )

    best_assortment = [int(round(x[i].X)) for i in range(n_products)]
    result = {
        "best_assortment": best_assortment,
        "best_assortment_str": assortment_to_string(best_assortment),
        "best_value": float(model.ObjVal),
        "solver_status": "OPTIMAL",
        "runtime_sec": float(model.Runtime),
        "num_types": n_types,
        "num_products": n_products,
        "objective_bound": float(model.ObjBound),
        "mip_gap": float(model.MIPGap),
        "policy": "true_mmnl",
        "type_probs": type_probs,
        "outside_utility": outside_utility,
    }
    return result


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve the optimal assortment exactly under the MMNL model."
    )
    parser.add_argument("--data", type=Path, required=True, help="Dataset JSON path.")
    parser.add_argument(
        "--time-limit-sec",
        type=float,
        default=None,
        help="Optional Gurobi time limit in seconds.",
    )
    parser.add_argument(
        "--mip-gap",
        type=float,
        default=0.0,
        help="Relative MIP gap tolerance. Use 0 for exact solves.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Optional Gurobi thread count.",
    )
    parser.add_argument(
        "--solver-verbose",
        action="store_true",
        help="Show Gurobi solver logs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    raw = load_assortment_json(args.data)
    if "meta" not in raw:
        raise ValueError("Dataset JSON must contain a 'meta' field.")
    meta = raw["meta"]

    start = time.monotonic()
    result = solve_mmnl_optimal_assortment(
        meta,
        time_limit_sec=args.time_limit_sec,
        mip_gap=args.mip_gap,
        solver_verbose=args.solver_verbose,
        threads=args.threads,
    )
    result["data"] = str(args.data)
    result["runtime_sec"] = float(time.monotonic() - start)
    result["time_limit_sec"] = args.time_limit_sec
    result["mip_gap_target"] = args.mip_gap
    result["threads"] = args.threads

    text = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
