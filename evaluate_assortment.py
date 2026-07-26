#!/usr/bin/env python3
"""Evaluate the expected reward of an assortment.

This script computes

    E[r(S, A) | S] = sum_a r(S, a) p(a | S)

where p(a | S) is either the true MNL policy stored in the dataset metadata
or a trained choice model checkpoint, and r(S, a) is either supplied by a
custom Python module or taken from the dataset metadata when reward depends
only on the chosen product.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Callable, Sequence

import torch

from assortment_dataset import load_assortment_json
from choice_models import GAsN, MNL, RAsN
from generate_assortment_data import build_mccm_model_from_meta


RewardFn = Callable[[Sequence[int], int, dict], float]


def parse_assortment(spec: str, n_products: int) -> list[int]:
    spec = spec.strip()
    if not spec:
        raise ValueError("assortment specification cannot be empty.")

    if "," in spec:
        values = [int(x.strip()) for x in spec.split(",") if x.strip()]
    else:
        if len(spec) != n_products:
            raise ValueError(
                f"Binary assortment string must have length {n_products}, got {len(spec)}."
            )
        values = [int(ch) for ch in spec]

    if len(values) != n_products:
        raise ValueError(f"Expected assortment length {n_products}, got {len(values)}.")
    if any(v not in (0, 1) for v in values):
        raise ValueError("Assortment entries must be 0 or 1.")
    return values


def mnl_probabilities(
    assortment: Sequence[int],
    utilities: Sequence[float],
    outside_utility: float,
) -> list[float]:
    offered = [i for i, bit in enumerate(assortment) if bit]
    logits = [outside_utility] + [utilities[i] for i in offered]
    max_logit = max(logits)
    exp_logits = [math.exp(x - max_logit) for x in logits]
    total = sum(exp_logits)
    probs = [x / total for x in exp_logits]

    full_probs = [0.0] * (len(utilities) + 1)
    full_probs[0] = probs[0]
    for pos, product_idx in enumerate(offered, start=1):
        full_probs[product_idx + 1] = probs[pos]
    return full_probs


def load_reward_module(path: str | Path) -> ModuleType:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_reward_fn(module_path: str | None) -> RewardFn:
    if module_path is None:
        def default_reward(S: Sequence[int], a: int, meta: dict) -> float:
            rewards = meta.get("rewards")
            if rewards is None:
                raise ValueError(
                    "Metadata does not contain 'rewards'; please provide --reward-module."
                )
            if a == 0:
                return float(meta.get("reward_no_purchase", 0.0))
            return float(rewards[a - 1])

        setattr(default_reward, "_is_default_reward_fn", True)
        return default_reward

    module = load_reward_module(module_path)
    if not hasattr(module, "reward"):
        raise AttributeError(
            f"{module_path} must define a callable named 'reward(S, a, meta)'."
        )
    reward_obj = getattr(module, "reward")
    if not callable(reward_obj):
        raise TypeError("'reward' must be callable.")

    sig = inspect.signature(reward_obj)
    if len(sig.parameters) not in (2, 3):
        raise TypeError(
            "reward must accept either (S, a) or (S, a, meta) as arguments."
        )

    def wrapped_reward(S: Sequence[int], a: int, meta: dict) -> float:
        if len(sig.parameters) == 3:
            return float(reward_obj(S, a, meta))
        return float(reward_obj(S, a))

    setattr(wrapped_reward, "_is_default_reward_fn", False)
    return wrapped_reward


def _ensure_batch_tensor(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if x.dim() == 1:
        return x.unsqueeze(0), True
    if x.dim() != 2:
        raise ValueError("Expected assortments to have shape [N] or [B, N].")
    return x, False


def batched_expected_reward(
    assortments: torch.Tensor,
    meta: dict,
    reward_fn: RewardFn,
    policy_kind: str,
    checkpoint_path: str | None,
    *,
    policy_model: torch.nn.Module | None = None,
) -> torch.Tensor:
    """Vectorized expected reward for a batch of assortments.

    When the reward function is the default metadata reward and the policy is
    true MNL, this is fully vectorized. Otherwise, it falls back to a Python
    loop over the batch.
    """
    assortments, squeezed = _ensure_batch_tensor(assortments)
    if policy_kind == "true_mnl" and getattr(reward_fn, "_is_default_reward_fn", False):
        utilities = meta.get("utilities")
        if utilities is None:
            raise ValueError("Metadata does not contain 'utilities'.")
        outside_utility = float(meta.get("outside_utility", 0.0))
        rewards = meta.get("rewards")
        if rewards is None:
            raise ValueError("Metadata does not contain 'rewards'.")
        reward_vec = torch.tensor(
            [float(meta.get("reward_no_purchase", 0.0))] + [float(x) for x in rewards],
            dtype=torch.float32,
            device=assortments.device,
        )
        utilities_t = torch.tensor(
            [float(u) for u in utilities], dtype=torch.float32, device=assortments.device
        )
        mask = assortments.to(dtype=torch.bool)
        product_logits = utilities_t.unsqueeze(0).expand(mask.shape[0], -1)
        neg_inf = torch.tensor(float("-inf"), device=assortments.device)
        product_logits = torch.where(mask, product_logits, neg_inf)
        outside_logits = torch.full(
            (mask.shape[0], 1), outside_utility, dtype=torch.float32, device=assortments.device
        )
        logits = torch.cat([outside_logits, product_logits], dim=-1)
        probs = torch.softmax(logits, dim=-1)
        values = (probs * reward_vec.unsqueeze(0)).sum(dim=-1)
        return values.squeeze(0) if squeezed else values

    if policy_kind == "true_mccm" and getattr(reward_fn, "_is_default_reward_fn", False):
        model = build_mccm_model_from_meta(meta, device=assortments.device)
        values = model.batched_expected_reward(assortments)
        return values.squeeze(0) if squeezed else values

    if policy_kind == "true_mmnl" and getattr(reward_fn, "_is_default_reward_fn", False):
        model = build_mmnl_model_from_meta(meta, device=assortments.device)
        values = model.batched_expected_reward(assortments)
        return values.squeeze(0) if squeezed else values

    if policy_kind == "model" and getattr(reward_fn, "_is_default_reward_fn", False):
        if policy_model is None:
            if checkpoint_path is None:
                raise ValueError("--checkpoint is required when --policy model is used.")
            policy_model = load_policy_from_checkpoint(checkpoint_path, device=assortments.device)
        probs = policy_model(assortments.float())
        rewards = meta.get("rewards")
        if rewards is None:
            raise ValueError("Metadata does not contain 'rewards'.")
        reward_vec = torch.tensor(
            [float(meta.get("reward_no_purchase", 0.0))] + [float(x) for x in rewards],
            dtype=torch.float32,
            device=assortments.device,
        )
        values = (probs * reward_vec.unsqueeze(0)).sum(dim=-1)
        return values.squeeze(0) if squeezed else values

    values = [
        expected_reward(assortment.tolist(), meta, reward_fn, policy_kind, checkpoint_path)
        for assortment in assortments
    ]
    values_t = torch.tensor(values, dtype=torch.float32, device=assortments.device)
    return values_t.squeeze(0) if squeezed else values_t


def build_batched_expected_reward_fn(
    meta: dict,
    reward_fn: RewardFn,
    policy_kind: str,
    checkpoint_path: str | None,
    *,
    device: torch.device | str = "cpu",
) -> Callable[[torch.Tensor], torch.Tensor]:
    device = torch.device(device)
    policy_model = None
    if policy_kind == "model" and checkpoint_path is not None:
        policy_model = load_policy_from_checkpoint(checkpoint_path, device=device)

    def scorer(assortments: torch.Tensor) -> torch.Tensor:
        if assortments.device != device:
            assortments = assortments.to(device)
        return batched_expected_reward(
            assortments,
            meta,
            reward_fn,
            policy_kind,
            checkpoint_path,
            policy_model=policy_model,
        )

    return scorer


def load_policy_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> torch.nn.Module:
    device = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_name = ckpt.get("model")
    n_products = int(ckpt["n_products"])
    hidden_sizes = ckpt.get("hidden_sizes")
    num_blocks = ckpt.get("num_blocks", 1)

    if model_name == "gasn":
        if hidden_sizes is None:
            hidden_sizes = [n_products]
        model = GAsN(n_products=n_products, hidden_sizes=hidden_sizes)
    elif model_name == "mnl":
        model = MNL(n_products=n_products)
    elif model_name == "rasn":
        num_blocks = int(num_blocks) if num_blocks is not None else 1
        state_dict = ckpt["model_state_dict"]
        # Older training checkpoints recorded a default hidden_sizes value even
        # though RAsN was instantiated in legacy linear-block mode. Infer that
        # architecture from its parameter keys so those checkpoints remain
        # loadable.
        legacy_rasn = any(
            key.startswith("blocks.") and key.endswith(".weight") and key.count(".") == 2
            for key in state_dict
        )
        if hidden_sizes is None or legacy_rasn:
            model = RAsN(n_products=n_products, num_blocks=num_blocks)
        else:
            model = RAsN(n_products=n_products, hidden_sizes=hidden_sizes, num_blocks=num_blocks)
    elif model_name == "tcn":
        tcn_config = dict(ckpt.get("tcn_config") or {})
        embedding_dim = int(tcn_config.get("embedding_dim", n_products))
        num_layers = int(tcn_config.get("num_layers", 1))
        num_heads = int(tcn_config.get("num_heads", 1))
        ffn_dim = int(tcn_config.get("ffn_dim", 4 * embedding_dim))
        dropout = float(tcn_config.get("dropout", 0.1))
        learnable_outside_logit = bool(tcn_config.get("learnable_outside_logit", True))
        model = TCNet(
            n_products=n_products,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            learnable_outside_logit=learnable_outside_logit,
        )
    else:
        raise ValueError(f"Unknown checkpoint model type {model_name!r}.")

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def policy_probabilities(
    assortment: Sequence[int],
    meta: dict,
    policy_kind: str,
    checkpoint_path: str | None,
) -> list[float]:
    if policy_kind == "true_mnl":
        utilities = meta.get("utilities")
        if utilities is None:
            raise ValueError("Metadata does not contain 'utilities'.")
        outside_utility = float(meta.get("outside_utility", 0.0))
        return mnl_probabilities(assortment, utilities, outside_utility)

    if policy_kind == "true_mmnl":
        model = build_mmnl_model_from_meta(meta)
        return [float(p) for p in model.choice_probabilities(assortment).tolist()]

    if policy_kind == "true_mccm":
        model = build_mccm_model_from_meta(meta)
        return [float(p) for p in model.choice_probabilities(assortment).tolist()]

    if policy_kind == "model":
        if checkpoint_path is None:
            raise ValueError("--checkpoint is required when --policy model is used.")
        model = load_policy_from_checkpoint(checkpoint_path)
        first_param = next(model.parameters(), None)
        device = first_param.device if first_param is not None else torch.device("cpu")
        S = torch.tensor(assortment, dtype=torch.float32, device=device).unsqueeze(0)
        probs = model(S).squeeze(0).tolist()
        return [float(p) for p in probs]

    raise ValueError("--policy must be one of: true_mnl, true_mmnl, true_mccm, model.")


def expected_reward(
    assortment: Sequence[int],
    meta: dict,
    reward_fn: RewardFn,
    policy_kind: str,
    checkpoint_path: str | None,
) -> float:
    probs = policy_probabilities(assortment, meta, policy_kind, checkpoint_path)
    if len(probs) != len(assortment) + 1:
        raise ValueError("Policy probability vector has unexpected length.")
    return float(
        sum(reward_fn(assortment, a, meta) * probs[a] for a in range(len(probs)))
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the expected reward of an assortment."
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="JSON dataset containing metadata for the true policy and rewards.",
    )
    parser.add_argument(
        "--assortment",
        type=str,
        required=True,
        help="Assortment as a binary string (e.g. 1010...) or comma-separated bits.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="true_mnl",
        choices=("true_mnl", "true_mmnl", "true_mccm", "model"),
        help="Use the true policy from metadata or a trained checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path when --policy model is used.",
    )
    parser.add_argument(
        "--reward-module",
        type=Path,
        default=None,
        help="Optional Python module with a reward(S, a, meta) function.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full result as JSON instead of a single number.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    raw = load_assortment_json(args.data)
    if "meta" not in raw:
        raise ValueError("Dataset JSON must contain a 'meta' field.")
    meta = raw["meta"]
    n_products = int(meta["N"])
    assortment = parse_assortment(args.assortment, n_products)
    reward_fn = build_reward_fn(str(args.reward_module) if args.reward_module else None)

    value = expected_reward(
        assortment=assortment,
        meta=meta,
        reward_fn=reward_fn,
        policy_kind=args.policy,
        checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
    )

    if args.json:
        result = {
            "assortment": assortment,
            "expected_reward": value,
            "policy": args.policy,
            "data": str(args.data),
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "reward_module": str(args.reward_module) if args.reward_module else None,
        }
        print(json.dumps(result, indent=2))
    else:
        print(value)


if __name__ == "__main__":
    main()
