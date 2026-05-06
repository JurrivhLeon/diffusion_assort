#!/usr/bin/env python3
"""Optimize an assortment under a trained neural choice model.

This script supports two backends:
- ``mip``: solver-backed optimization using Gurobi.
- ``exhaustive``: exact enumeration reference implementation.

The MIP backend follows the Section 5.1 formulation from the paper in spirit:
it encodes the ReLU network with big-M constraints, approximates the final
``exp`` terms with SOS2 piecewise-linear interpolation, and solves the resulting
fractional objective with Dinkelbach iterations.

The solver-backed backend currently assumes the reward depends only on the
chosen option, which matches the generated dataset in this repository.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Sequence

import torch
import gurobipy as gp
from tqdm.auto import tqdm

from evaluate_assortment import load_assortment_json, load_policy_from_checkpoint
from exhaustive_search_assortment import (
    build_policy_kind,
    count_assortments,
    iter_assortments,
)


class OptimizationStatus:
    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    NO_SOLUTION_FOUND = "NO_SOLUTION_FOUND"
    INFEASIBLE = "INFEASIBLE"
    INF_OR_UNBD = "INF_OR_UNBD"
    UNBOUNDED = "UNBOUNDED"
    ERROR = "ERROR"
    LOADED = "LOADED"
    OTHER = "OTHER"
    CUTOFF = "CUTOFF"
    INT_INFEASIBLE = "INT_INFEASIBLE"


CBC = "GUROBI"
BINARY = gp.GRB.BINARY
CONTINUOUS = gp.GRB.CONTINUOUS
MAXIMIZE = gp.GRB.MAXIMIZE
MINIMIZE = gp.GRB.MINIMIZE


def xsum(terms):
    return gp.quicksum(terms)


def _var_value(var) -> float:
    if hasattr(var, "X"):
        return float(var.X)
    return float(var.x)


class Model:
    def __init__(self, name: str = "", sense: str = "MIN", solver_name: str = ""):
        self._model = gp.Model(name=name)
        self._model.Params.OutputFlag = 0
        self._sense = sense

    def add_var(self, lb: float = -gp.GRB.INFINITY, ub: float = gp.GRB.INFINITY, var_type=CONTINUOUS, name: str = ""):
        return self._model.addVar(lb=lb, ub=ub, vtype=var_type, name=name)

    def add_sos(self, sos, sos_type):
        vars_, wts = zip(*sos)
        sos_kind = gp.GRB.SOS_TYPE2 if int(sos_type) == 2 else gp.GRB.SOS_TYPE1
        self._model.addSOS(sos_kind, list(vars_), list(wts))

    def __iadd__(self, constr):
        self._model.addConstr(constr)
        return self

    @property
    def verbose(self) -> int:
        return int(self._model.Params.OutputFlag)

    @verbose.setter
    def verbose(self, value: int) -> None:
        self._model.Params.OutputFlag = int(bool(value))

    @property
    def objective(self):
        return self._model.getObjective()

    @objective.setter
    def objective(self, expr) -> None:
        self._model.setObjective(expr, self._sense)

    @property
    def start(self):
        raise AttributeError("Use assignment to set the MIP start.")

    @start.setter
    def start(self, start_values):
        for var, val in start_values:
            var.Start = val

    def optimize(
        self,
        max_seconds: float = math.inf,
        max_nodes: int = 1 << 30,
        max_solutions: int = 1 << 30,
        max_seconds_same_incumbent: float = math.inf,
        max_nodes_same_incumbent: int = 1 << 30,
        relax: bool = False,
    ):
        del max_nodes, max_solutions, max_seconds_same_incumbent, max_nodes_same_incumbent, relax
        if math.isfinite(max_seconds):
            self._model.Params.TimeLimit = float(max_seconds)
        self._model.optimize()
        if self._model.Status == gp.GRB.OPTIMAL:
            return OptimizationStatus.OPTIMAL
        if self._model.SolCount > 0:
            return OptimizationStatus.FEASIBLE
        if self._model.Status == gp.GRB.INFEASIBLE:
            return OptimizationStatus.INFEASIBLE
        if self._model.Status == gp.GRB.UNBOUNDED:
            return OptimizationStatus.UNBOUNDED
        if self._model.Status == gp.GRB.INF_OR_UNBD:
            return OptimizationStatus.INF_OR_UNBD
        if self._model.Status == gp.GRB.LOADED:
            return OptimizationStatus.LOADED
        if self._model.Status == gp.GRB.CUTOFF:
            return OptimizationStatus.CUTOFF
        if self._model.Status == gp.GRB.NUMERIC:
            return OptimizationStatus.ERROR
        return OptimizationStatus.NO_SOLUTION_FOUND


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize an assortment under a trained neural choice model."
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="JSON dataset containing metadata for rewards and model dimensions.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Trained choice-model checkpoint (.pt). Required when --policy model is used.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="model",
        choices=("model", "true_mnl"),
        help="Use the trained model or the true MNL policy from metadata.",
    )
    parser.add_argument(
        "--reward-module",
        type=Path,
        default=None,
        help="Optional Python module with a reward(S, a, meta) function. Supported in exhaustive mode only.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="mip",
        choices=("mip", "exhaustive"),
        help="Use the solver-backed MIP formulation or exact enumeration.",
    )
    parser.add_argument(
        "--exp-pieces",
        type=int,
        default=8,
        help="Number of linear segments used to approximate exp in the MIP backend.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=math.inf,
        help="Total wall-clock time limit in seconds for the MIP backend; use inf for no limit.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Maximum Dinkelbach iterations for the MIP backend.",
    )
    parser.add_argument(
        "--dinkelbach-tol",
        type=float,
        default=1e-3,
        help="Stopping tolerance for the fractional objective iterations.",
    )
    parser.add_argument(
        "--solver-verbose",
        action="store_true",
        help="Show Gurobi solver logs.",
    )
    parser.add_argument(
        "--max-cardinality",
        type=int,
        default=None,
        help="Optional upper bound on assortment size.",
    )
    parser.add_argument(
        "--exact-cardinality",
        type=int,
        default=None,
        help="If set, search only assortments of this cardinality.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the result as JSON.",
    )
    return parser


def _validate_cardinality_args(
    n_products: int,
    max_cardinality: int | None,
    exact_cardinality: int | None,
) -> None:
    if max_cardinality is not None and max_cardinality < 0:
        raise ValueError("--max-cardinality must be non-negative.")
    if exact_cardinality is not None and exact_cardinality < 0:
        raise ValueError("--exact-cardinality must be non-negative.")
    if max_cardinality is not None and max_cardinality > n_products:
        raise ValueError("--max-cardinality cannot exceed N.")
    if exact_cardinality is not None and exact_cardinality > n_products:
        raise ValueError("--exact-cardinality cannot exceed N.")
    if max_cardinality is not None and exact_cardinality is not None:
        raise ValueError("Specify at most one of --max-cardinality or --exact-cardinality.")


def _reward_vector(meta: dict) -> list[float]:
    rewards = meta.get("rewards")
    if rewards is None:
        raise ValueError("Metadata does not contain 'rewards'.")
    reward_no_purchase = float(meta.get("reward_no_purchase", 0.0))
    return [reward_no_purchase] + [float(x) for x in rewards]


def _heuristic_start_assortment(
    reward_coeffs: Sequence[float],
    max_cardinality: int | None,
    exact_cardinality: int | None,
) -> list[int]:
    if exact_cardinality is not None:
        product_rewards = [float(x) for x in reward_coeffs[1:]]
        n_products = len(product_rewards)
        if n_products == 0:
            return []
        ranked = sorted(range(n_products), key=lambda j: product_rewards[j], reverse=True)
        k = min(int(exact_cardinality), n_products)
        chosen = ranked[:k]
    else:
        # For the sparse unconstrained baseline, start from the all-zero assortment.
        # This keeps the initial incumbent simple and feasible for max-cardinality
        # or unconstrained runs.
        return [0] * (len(reward_coeffs) - 1)

    assortment = [0] * len(product_rewards)
    for j in chosen:
        assortment[j] = 1
    return assortment


def _affine_bounds(
    weight: Sequence[Sequence[float]],
    bias: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> tuple[list[float], list[float]]:
    out_lower: list[float] = []
    out_upper: list[float] = []
    for row, b in zip(weight, bias):
        lo = float(b)
        hi = float(b)
        for w, l, u in zip(row, lower, upper):
            w = float(w)
            if w >= 0:
                lo += w * float(l)
                hi += w * float(u)
            else:
                lo += w * float(u)
                hi += w * float(l)
        out_lower.append(lo)
        out_upper.append(hi)
    return out_lower, out_upper


def _relu_bounds(lower: Sequence[float], upper: Sequence[float]) -> tuple[list[float], list[float]]:
    return [max(0.0, float(x)) for x in lower], [max(0.0, float(x)) for x in upper]


def _extract_layer_params(layer: torch.nn.Linear) -> tuple[list[list[float]], list[float]]:
    weight = layer.weight.detach().cpu().tolist()
    bias = layer.bias.detach().cpu().tolist()
    return weight, bias


def _build_relu_block(
    model: Model,
    input_vars: Sequence,
    weight: Sequence[Sequence[float]],
    bias: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    prefix: str,
) -> list:
    output_vars: list = []
    in_dim = len(input_vars)
    for idx, (row, b, lo, hi) in enumerate(zip(weight, bias, lower, upper)):
        pre = model.add_var(
            lb=float(lo),
            ub=float(hi),
            var_type=CONTINUOUS,
            name=f"{prefix}_pre_{idx}",
        )
        model += pre == xsum(float(row[j]) * input_vars[j] for j in range(in_dim)) + float(b)

        if hi <= 0.0:
            post = model.add_var(lb=0.0, ub=0.0, var_type=CONTINUOUS, name=f"{prefix}_post_{idx}")
            model += post == 0
        elif lo >= 0.0:
            post = pre
        else:
            post = model.add_var(
                lb=0.0,
                ub=float(hi),
                var_type=CONTINUOUS,
                name=f"{prefix}_post_{idx}",
            )
            active = model.add_var(var_type=BINARY, name=f"{prefix}_active_{idx}")
            model += post >= pre
            model += post >= 0
            model += post <= pre - float(lo) * (1 - active)
            model += post <= float(hi) * active
        output_vars.append(post)
    return output_vars


def _piecewise_exp(
    model: Model,
    x_var,
    lower: float,
    upper: float,
    num_pieces: int,
    prefix: str,
) -> tuple:
    if upper <= lower + 1e-12:
        value = math.exp(lower)
        y = model.add_var(lb=value, ub=value, var_type=CONTINUOUS, name=f"{prefix}_exp")
        model += y == value
        model += x_var == float(lower)
        return y

    num_points = max(2, int(num_pieces) + 1)
    xs = [lower + (upper - lower) * k / (num_points - 1) for k in range(num_points)]
    ys = [math.exp(x) for x in xs]
    lambdas = [
        model.add_var(lb=0.0, ub=1.0, var_type=CONTINUOUS, name=f"{prefix}_lambda_{k}")
        for k in range(num_points)
    ]
    y = model.add_var(
        lb=ys[0],
        ub=ys[-1],
        var_type=CONTINUOUS,
        name=f"{prefix}_exp",
    )
    model += x_var == xsum(xs[k] * lambdas[k] for k in range(num_points))
    model += y == xsum(ys[k] * lambdas[k] for k in range(num_points))
    model += xsum(lambdas) == 1
    model.add_sos(list(zip(lambdas, range(num_points))), 2)
    return y


def _load_choice_model(checkpoint_path: Path):
    return load_policy_from_checkpoint(checkpoint_path)


def _build_true_mnl_mip(
    meta: dict,
    max_cardinality: int | None,
    exact_cardinality: int | None,
) -> tuple[Model, list, list, list[float], float]:
    n_products = int(meta["N"])
    utilities = meta.get("utilities")
    if utilities is None:
        raise ValueError("Metadata does not contain 'utilities' required for true MNL.")
    outside_utility = float(meta.get("outside_utility", 0.0))
    reward_coeffs = _reward_vector(meta)

    exp_outside = math.exp(outside_utility)
    exp_utils = [math.exp(float(u)) for u in utilities]

    model = Model(sense=MAXIMIZE, solver_name=CBC)
    assort_vars = [
        model.add_var(var_type=BINARY, name=f"s_{j}")
        for j in range(n_products)
    ]
    if exact_cardinality is not None:
        model += xsum(assort_vars) == int(exact_cardinality)
    elif max_cardinality is not None:
        model += xsum(assort_vars) <= int(max_cardinality)

    offer_vars = []
    for j in range(n_products):
        y = model.add_var(
            lb=0.0,
            ub=float(exp_utils[j]),
            var_type=CONTINUOUS,
            name=f"offer_{j}",
        )
        model += y == float(exp_utils[j]) * assort_vars[j]
        offer_vars.append(y)

    return model, assort_vars, offer_vars, reward_coeffs, exp_outside


def _build_learned_mnl_mip(
    meta: dict,
    checkpoint_path: Path,
    max_cardinality: int | None,
    exact_cardinality: int | None,
) -> tuple[Model, list, list, list[float], float, torch.nn.Module]:
    model = _load_choice_model(checkpoint_path)
    if model.__class__.__name__ != "MNL":
        raise ValueError(
            f"Expected an MNL checkpoint, got {model.__class__.__name__!r}."
        )
    n_products = int(meta["N"])
    if int(getattr(model, "n_products", n_products)) != n_products:
        raise ValueError("Checkpoint and dataset disagree on N.")

    reward_coeffs = _reward_vector(meta)
    outside_logit = float(model.outside_logit.detach().cpu().item())
    product_utilities = [float(x) for x in model.product_utilities.detach().cpu().tolist()]
    exp_outside = math.exp(outside_logit)
    exp_utils = [math.exp(x) for x in product_utilities]

    mip_model = Model(sense=MAXIMIZE, solver_name=CBC)
    assort_vars = [
        mip_model.add_var(var_type=BINARY, name=f"s_{j}")
        for j in range(n_products)
    ]
    if exact_cardinality is not None:
        mip_model += xsum(assort_vars) == int(exact_cardinality)
    elif max_cardinality is not None:
        mip_model += xsum(assort_vars) <= int(max_cardinality)

    offer_vars = []
    for j in range(n_products):
        y = mip_model.add_var(
            lb=0.0,
            ub=float(exp_utils[j]),
            var_type=CONTINUOUS,
            name=f"offer_{j}",
        )
        mip_model += y == float(exp_utils[j]) * assort_vars[j]
        offer_vars.append(y)

    return mip_model, assort_vars, offer_vars, reward_coeffs, exp_outside, model


def _build_neural_mip(
    meta: dict,
    checkpoint_path: Path,
    max_cardinality: int | None,
    exact_cardinality: int | None,
    exp_pieces: int,
) -> tuple[Model, list, list, list[float], float, torch.nn.Module]:
    model = _load_choice_model(checkpoint_path)
    reward_coeffs = _reward_vector(meta)
    n_products = int(meta["N"])
    if int(getattr(model, "n_products", n_products)) != n_products:
        raise ValueError("Checkpoint and dataset disagree on N.")

    if hasattr(model, "outside_logit"):
        outside_logit = float(model.outside_logit.detach().cpu().item())
    else:
        outside_logit = 0.0
    exp_outside = math.exp(outside_logit)

    input_lower = [0.0] * n_products
    input_upper = [1.0] * n_products
    layer_specs: list[dict] = []

    if model.__class__.__name__ == "GAsN":
        current_lower = input_lower
        current_upper = input_upper
        for idx, layer in enumerate(model.layers):
            weight, bias = _extract_layer_params(layer)
            pre_lower, pre_upper = _affine_bounds(weight, bias, current_lower, current_upper)
            post_lower, post_upper = _relu_bounds(pre_lower, pre_upper)
            layer_specs.append(
                {
                    "kind": "relu",
                    "weight": weight,
                    "bias": bias,
                    "pre_lower": pre_lower,
                    "pre_upper": pre_upper,
                    "post_lower": post_lower,
                    "post_upper": post_upper,
                    "name": f"layer_{idx}",
                }
            )
            current_lower, current_upper = post_lower, post_upper
        logits_lower = current_lower
        logits_upper = current_upper
        model_kind = "gasn"
    elif model.__class__.__name__ == "RAsN":
        current_lower = input_lower
        current_upper = input_upper
        for idx, block in enumerate(model.blocks):
            weight, bias = _extract_layer_params(block)
            pre_lower, pre_upper = _affine_bounds(weight, bias, current_lower, current_upper)
            relu_lower, relu_upper = _relu_bounds(pre_lower, pre_upper)
            next_lower = [float(a) + float(b) for a, b in zip(current_lower, relu_lower)]
            next_upper = [float(a) + float(b) for a, b in zip(current_upper, relu_upper)]
            layer_specs.append(
                {
                    "kind": "residual",
                    "weight": weight,
                    "bias": bias,
                    "pre_lower": pre_lower,
                    "pre_upper": pre_upper,
                    "relu_lower": relu_lower,
                    "relu_upper": relu_upper,
                    "post_lower": next_lower,
                    "post_upper": next_upper,
                    "name": f"block_{idx}",
                }
            )
            current_lower, current_upper = next_lower, next_upper
        logits_lower = current_lower
        logits_upper = current_upper
        model_kind = "rasn"
    else:
        raise ValueError(
            f"Unsupported checkpoint model type {model.__class__.__name__!r} for MIP backend."
        )

    mip_model = Model(sense=MAXIMIZE, solver_name=CBC)
    assort_vars = [
        mip_model.add_var(var_type=BINARY, name=f"s_{j}")
        for j in range(n_products)
    ]
    if exact_cardinality is not None:
        mip_model += xsum(assort_vars) == int(exact_cardinality)
    elif max_cardinality is not None:
        mip_model += xsum(assort_vars) <= int(max_cardinality)

    current_vars = assort_vars
    if model_kind == "gasn":
        for spec in layer_specs:
            current_vars = _build_relu_block(
                mip_model,
                current_vars,
                spec["weight"],
                spec["bias"],
                spec["pre_lower"],
                spec["pre_upper"],
                spec["name"],
            )
    else:
        for spec in layer_specs:
            pre_vars = []
            for idx, (row, b, lo, hi) in enumerate(
                zip(spec["weight"], spec["bias"], spec["pre_lower"], spec["pre_upper"])
            ):
                pre = mip_model.add_var(
                    lb=float(lo),
                    ub=float(hi),
                    var_type=CONTINUOUS,
                    name=f"{spec['name']}_pre_{idx}",
                )
                mip_model += pre == xsum(
                    float(row[j]) * current_vars[j] for j in range(n_products)
                ) + float(b)
                pre_vars.append(pre)

            relu_vars: list = []
            for idx, (pre, pre_lo, pre_hi, relu_lo, relu_hi) in enumerate(
                zip(
                    pre_vars,
                    spec["pre_lower"],
                    spec["pre_upper"],
                    spec["relu_lower"],
                    spec["relu_upper"],
                )
            ):
                if pre_hi <= 0.0:
                    relu_var = mip_model.add_var(
                        lb=0.0,
                        ub=0.0,
                        var_type=CONTINUOUS,
                        name=f"{spec['name']}_relu_{idx}",
                    )
                    mip_model += relu_var == 0
                elif pre_lo >= 0.0:
                    relu_var = pre
                else:
                    relu_var = mip_model.add_var(
                        lb=0.0,
                        ub=float(relu_hi),
                        var_type=CONTINUOUS,
                        name=f"{spec['name']}_relu_{idx}",
                    )
                    active = mip_model.add_var(
                        var_type=BINARY,
                        name=f"{spec['name']}_active_{idx}",
                    )
                    mip_model += relu_var >= pre
                    mip_model += relu_var >= 0
                    mip_model += relu_var <= pre - float(pre_lo) * (1 - active)
                    mip_model += relu_var <= float(pre_hi) * active
                relu_vars.append(relu_var)

            next_vars: list = []
            for idx, (prev, relu_var, lo, hi) in enumerate(
                zip(current_vars, relu_vars, spec["post_lower"], spec["post_upper"])
            ):
                out = mip_model.add_var(
                    lb=float(lo),
                    ub=float(hi),
                    var_type=CONTINUOUS,
                    name=f"{spec['name']}_out_{idx}",
                )
                mip_model += out == prev + relu_var
                next_vars.append(out)
            current_vars = next_vars

    logits_vars = current_vars

    offer_vars: list = []
    if model_kind == "gasn":
        for j, (logit_var, lo, hi) in enumerate(zip(logits_vars, logits_lower, logits_upper)):
            if hi > 50.0:
                raise ValueError(
                    "Encountered an excessively large logit upper bound; the exp approximation "
                    "would be numerically unstable. Consider reducing the model scale or using "
                    "a different checkpoint."
                )
            exp_hi = math.exp(float(hi))
            exp_var = _piecewise_exp(
                mip_model,
                logit_var,
                float(lo),
                float(hi),
                exp_pieces,
                prefix=f"exp_{j}",
            )
            offer = mip_model.add_var(
                lb=0.0,
                ub=exp_hi,
                var_type=CONTINUOUS,
                name=f"offer_{j}",
            )
            mip_model += offer <= exp_var
            mip_model += offer <= exp_hi * assort_vars[j]
            mip_model += offer >= exp_var - exp_hi * (1 - assort_vars[j])
            mip_model += offer >= 0
            offer_vars.append(offer)
    elif model_kind == "rasn":
        # The RAsN branch reuses the same final logits. The build above already
        # produced the correct output variables, so the exp linearization is the
        # same as for GAsN.
        for j, (logit_var, lo, hi) in enumerate(zip(logits_vars, logits_lower, logits_upper)):
            if hi > 50.0:
                raise ValueError(
                    "Encountered an excessively large logit upper bound; the exp approximation "
                    "would be numerically unstable. Consider reducing the model scale or using "
                    "a different checkpoint."
                )
            exp_hi = math.exp(float(hi))
            exp_var = _piecewise_exp(
                mip_model,
                logit_var,
                float(lo),
                float(hi),
                exp_pieces,
                prefix=f"exp_{j}",
            )
            offer = mip_model.add_var(
                lb=0.0,
                ub=exp_hi,
                var_type=CONTINUOUS,
                name=f"offer_{j}",
            )
            mip_model += offer <= exp_var
            mip_model += offer <= exp_hi * assort_vars[j]
            mip_model += offer >= exp_var - exp_hi * (1 - assort_vars[j])
            mip_model += offer >= 0
            offer_vars.append(offer)
    else:  # pragma: no cover - guarded above
        raise AssertionError(model_kind)

    return mip_model, assort_vars, offer_vars, reward_coeffs, exp_outside, model


def _solve_fractional_mip(
    mip_model: Model,
    assort_vars: Sequence,
    offer_vars: Sequence,
    reward_coeffs: Sequence[float],
    exp_outside: float,
    *,
    time_limit: float,
    max_iterations: int,
    tol: float,
    solver_verbose: bool,
) -> tuple[list[int], float, float, int, str]:
    mip_model.verbose = 1 if solver_verbose else 0
    reward_outside = float(reward_coeffs[0])
    product_rewards = [float(x) for x in reward_coeffs[1:]]
    lambda_val = 0.0
    best_assortment: list[int] | None = None
    best_exact_value = float("-inf")
    best_mip_value = float("-inf")
    best_gap = float("inf")
    status_text = "not_started"
    start = time.monotonic()
    iterations_done = 0

    with tqdm(total=max_iterations, unit="iter", desc="Dinkelbach") as pbar:
        for iteration in range(max_iterations):
            elapsed = time.monotonic() - start
            remaining = max(0.0, time_limit - elapsed)
            if remaining <= 0.0:
                status_text = "time_limit"
                break

            objective = xsum((product_rewards[j] - lambda_val) * offer_vars[j] for j in range(len(offer_vars)))
            mip_model.objective = objective
            status = mip_model.optimize(max_seconds=remaining)
            status_text = str(status)

            if status not in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
                raise RuntimeError(f"MIP solve failed with status {status}.")

            assortment = [1 if _var_value(var) >= 0.5 else 0 for var in assort_vars]
            numerator = reward_outside * exp_outside + sum(
                product_rewards[j] * _var_value(offer_vars[j]) for j in range(len(offer_vars))
            )
            denominator = exp_outside + sum(_var_value(offer_vars[j]) for j in range(len(offer_vars)))
            if denominator <= 0:
                raise RuntimeError("MIP produced a non-positive denominator, which should be impossible.")

            exact_mip_objective = numerator / denominator
            gap = numerator - lambda_val * denominator
            new_lambda = numerator / denominator

            if exact_mip_objective > best_exact_value:
                best_exact_value = exact_mip_objective
                best_assortment = assortment
                best_mip_value = exact_mip_objective
                best_gap = gap

            pbar.set_postfix(
                {
                    "approx_obj": f"{exact_mip_objective:.6f}",
                    "gap": f"{gap:.3e}",
                    "lambda": f"{lambda_val:.6f}",
                }
            )
            pbar.update(1)
            iterations_done = iteration + 1

            if abs(gap) <= tol or abs(new_lambda - lambda_val) <= tol:
                lambda_val = new_lambda
                break
            lambda_val = new_lambda

    if best_assortment is None:
        raise RuntimeError("MIP backend did not find a feasible assortment.")

    return best_assortment, best_exact_value, best_mip_value, iterations_done, status_text


def _evaluate_selected_assortment(
    assortment: Sequence[int],
    meta: dict,
    policy_kind: str,
    checkpoint_path: Path | None,
    model: torch.nn.Module | None = None,
) -> float:
    reward_coeffs = _reward_vector(meta)
    if policy_kind == "true_mnl":
        utilities = meta.get("utilities")
        if utilities is None:
            raise ValueError("Metadata does not contain 'utilities'.")
        outside_utility = float(meta.get("outside_utility", 0.0))
        logits = [outside_utility] + [float(utilities[i]) for i in range(len(utilities))]
        offered = [i for i, bit in enumerate(assortment) if bit]
        chosen_logits = [logits[0]] + [logits[i + 1] for i in offered]
        max_logit = max(chosen_logits)
        exp_logits = [math.exp(x - max_logit) for x in chosen_logits]
        total = sum(exp_logits)
        probs = [0.0] * (len(assortment) + 1)
        probs[0] = exp_logits[0] / total
        for pos, product_idx in enumerate(offered, start=1):
            probs[product_idx + 1] = exp_logits[pos] / total
        return float(sum(reward_coeffs[a] * probs[a] for a in range(len(probs))))

    if model is None:
        if checkpoint_path is None:
            raise ValueError("--checkpoint is required when evaluating a trained model.")
        model = _load_choice_model(checkpoint_path)
    S = torch.tensor(assortment, dtype=torch.float32).unsqueeze(0)
    probs = model(S).squeeze(0).tolist()
    return float(sum(reward_coeffs[a] * probs[a] for a in range(len(probs))))


def _solve_mip_backend(
    meta: dict,
    checkpoint_path: Path | None,
    policy_kind: str,
    max_cardinality: int | None,
    exact_cardinality: int | None,
    exp_pieces: int,
    time_limit: float,
    max_iterations: int,
    tol: float,
    solver_verbose: bool,
) -> dict:
    if policy_kind == "true_mnl":
        mip_model, assort_vars, offer_vars, reward_coeffs, exp_outside = _build_true_mnl_mip(
            meta,
            max_cardinality=max_cardinality,
            exact_cardinality=exact_cardinality,
        )
        model_obj = None
    else:
        if checkpoint_path is None:
            raise ValueError("--checkpoint is required when --policy model is used.")
        loaded_model = _load_choice_model(checkpoint_path)
        if loaded_model.__class__.__name__ == "MNL":
            mip_model, assort_vars, offer_vars, reward_coeffs, exp_outside, model_obj = _build_learned_mnl_mip(
                meta,
                checkpoint_path=checkpoint_path,
                max_cardinality=max_cardinality,
                exact_cardinality=exact_cardinality,
            )
        else:
            mip_model, assort_vars, offer_vars, reward_coeffs, exp_outside, model_obj = _build_neural_mip(
                meta,
                checkpoint_path=checkpoint_path,
                max_cardinality=max_cardinality,
                exact_cardinality=exact_cardinality,
                exp_pieces=exp_pieces,
            )

    start_assortment = _heuristic_start_assortment(
        reward_coeffs,
        max_cardinality=max_cardinality,
        exact_cardinality=exact_cardinality,
    )
    mip_model.start = [
        (assort_vars[j], 1)
        for j, bit in enumerate(start_assortment)
        if bit
    ]

    best_assortment, approx_value, mip_value, iterations, status_text = _solve_fractional_mip(
        mip_model,
        assort_vars,
        offer_vars,
        reward_coeffs,
        exp_outside,
        time_limit=time_limit,
        max_iterations=max_iterations,
        tol=tol,
        solver_verbose=solver_verbose,
    )
    exact_value = _evaluate_selected_assortment(
        best_assortment,
        meta,
        policy_kind=policy_kind,
        checkpoint_path=checkpoint_path,
        model=model_obj,
    )
    return {
        "best_assortment": best_assortment,
        "best_value": exact_value,
        "mip_value": mip_value,
        "approx_value": approx_value,
        "iterations": iterations,
        "solver_status": status_text,
        "backend": "mip",
    }


def _solve_exhaustive_backend(
    args: argparse.Namespace,
    meta: dict,
    n_products: int,
) -> dict:
    reward_fn = None
    if args.reward_module is None:
        reward_fn = lambda S, a, m: _reward_vector(m)[a]  # noqa: E731
    else:
        from evaluate_assortment import build_reward_fn

        reward_fn = build_reward_fn(str(args.reward_module))

    policy_fn = build_policy_kind(meta, args.policy, str(args.checkpoint) if args.checkpoint else None)
    total = count_assortments(
        n_products,
        max_cardinality=args.max_cardinality,
        exact_cardinality=args.exact_cardinality,
    )

    best_S: list[int] | None = None
    best_value = float("-inf")
    num_scored = 0

    with tqdm(total=total, unit="assortment", desc="Exhaustive") as pbar:
        for S in iter_assortments(
            n_products,
            max_cardinality=args.max_cardinality,
            exact_cardinality=args.exact_cardinality,
        ):
            probs = policy_fn(S)
            if len(probs) != n_products + 1:
                raise ValueError("Policy probability vector has unexpected length.")
            value = float(sum(reward_fn(S, a, meta) * probs[a] for a in range(len(probs))))
            num_scored += 1
            pbar.update(1)
            if value > best_value:
                best_value = value
                best_S = list(S)

    if best_S is None:
        raise RuntimeError("No assortments were scored.")

    return {
        "best_assortment": best_S,
        "best_value": best_value,
        "num_scored": num_scored,
        "backend": "exhaustive",
    }


def main() -> None:
    args = build_argparser().parse_args()
    raw = load_assortment_json(args.data)
    if "meta" not in raw:
        raise ValueError("Dataset JSON must contain a 'meta' field.")
    meta = raw["meta"]
    n_products = int(meta["N"])

    _validate_cardinality_args(n_products, args.max_cardinality, args.exact_cardinality)

    if args.backend == "mip" and args.reward_module is not None:
        raise NotImplementedError(
            "The solver-backed MIP backend currently assumes the reward depends only on the choice "
            "and is read from the dataset metadata. Use --backend exhaustive for custom reward modules."
        )

    start = time.monotonic()

    if args.backend == "mip":
        result = _solve_mip_backend(
            meta=meta,
            checkpoint_path=args.checkpoint,
            policy_kind=args.policy,
            max_cardinality=args.max_cardinality,
            exact_cardinality=args.exact_cardinality,
            exp_pieces=args.exp_pieces,
            time_limit=args.time_limit,
            max_iterations=args.max_iterations,
            tol=args.dinkelbach_tol,
            solver_verbose=args.solver_verbose,
        )
    else:
        result = _solve_exhaustive_backend(args, meta, n_products)

    result.update(
        {
            "policy": args.policy,
            "data": str(args.data),
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "reward_module": str(args.reward_module) if args.reward_module else None,
            "max_cardinality": args.max_cardinality,
            "exact_cardinality": args.exact_cardinality,
            "runtime_sec": time.monotonic() - start,
        }
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"backend: {result['backend']}")
        print(f"best value: {result['best_value']}")
        print(f"best assortment: {''.join(str(x) for x in result['best_assortment'])}")
        if "iterations" in result:
            print(f"iterations: {result['iterations']}")
            print(f"solver status: {result.get('solver_status')}")
            print(f"mip value: {result.get('mip_value')}")
        if "num_scored" in result:
            print(f"num scored: {result['num_scored']}")


if __name__ == "__main__":
    main()
