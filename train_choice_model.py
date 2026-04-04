#!/usr/bin/env python3
"""Train a feature-free neural choice model with maximum likelihood."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, random_split

from assortment_dataset import AssortmentChoiceDataset
from choice_models import GAsN, RAsN


@dataclass
class TrainResult:
    epoch: int
    train_nll: float
    val_nll: float
    val_accuracy: float


def parse_sizes(spec: str | None, default: Sequence[int]) -> list[int]:
    if spec is None or not spec.strip():
        return list(default)
    return [int(x) for x in spec.split(",") if x.strip()]


def build_model(
    model_name: str,
    n_products: int,
    hidden_sizes: Sequence[int],
    num_blocks: int,
) -> torch.nn.Module:
    if model_name == "gasn":
        return GAsN(n_products=n_products, hidden_sizes=hidden_sizes)
    if model_name == "rasn":
        if num_blocks <= 0:
            raise ValueError("--num-blocks must be positive.")
        return RAsN(n_products=n_products, num_blocks=num_blocks)
    raise ValueError("--model must be one of: gasn, rasn.")


def build_run_config(
    *,
    args: argparse.Namespace,
    n_products: int,
    hidden_sizes: Sequence[int],
) -> dict:
    architecture = {
        "name": args.model,
        "n_products": n_products,
        "hidden_sizes": list(hidden_sizes) if args.model == "gasn" else None,
        "num_blocks": args.num_blocks if args.model == "rasn" else None,
        "learnable_outside_logit": True,
    }
    training = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "device": args.device,
        "patience": args.patience,
    }
    data = {
        "path": str(args.data),
        "n_products": n_products,
    }
    return {"architecture": architecture, "training": training, "data": data}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_nll = 0.0
    total_examples = 0
    total_correct = 0.0

    for batch in loader:
        S = batch["S"].to(device)
        A = batch["A"].to(device)
        probs = model(S)
        nll = model.negative_log_likelihood(S, A, reduction="sum")
        total_nll += float(nll.item())
        total_examples += int(A.shape[0])
        total_correct += float((probs.argmax(dim=-1) == A).float().sum().item())

    if total_examples == 0:
        raise ValueError("Loader produced no examples.")

    return total_nll / total_examples, total_correct / total_examples


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_nll = 0.0
    total_examples = 0

    for batch in loader:
        S = batch["S"].to(device)
        A = batch["A"].to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = model.negative_log_likelihood(S, A, reduction="mean")
        loss.backward()
        optimizer.step()

        batch_size = int(A.shape[0])
        total_nll += float(loss.item()) * batch_size
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("Loader produced no examples.")

    return total_nll / total_examples


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train GAsN or RAsN on assortment-choice JSON data."
    )
    parser.add_argument("--data", type=Path, required=True, help="Path to JSON dataset.")
    parser.add_argument(
        "--model",
        type=str,
        default="gasn",
        choices=("gasn", "rasn"),
        help="Choice model architecture.",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=str,
        default=None,
        help="Comma-separated hidden sizes for GAsN, e.g. '64,64'. Default: 'N'.",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=1,
        help="Number of residual blocks for RAsN.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Adam weight decay.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of data held out for validation.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Training device, e.g. 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/choice_model"),
        help="Directory for checkpoints and metrics.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early-stopping patience on validation NLL.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.patience < 0:
        raise ValueError("--patience must be non-negative.")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = AssortmentChoiceDataset(args.data)
    n_products = dataset.n_products
    hidden_sizes = parse_sizes(args.hidden_sizes, default=(n_products,))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    n_val = max(1, int(round(len(dataset) * args.val_fraction)))
    n_train = len(dataset) - n_val
    if n_train <= 0:
        raise ValueError("Not enough data after validation split.")

    gen = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = build_model(args.model, n_products, hidden_sizes, args.num_blocks).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    run_config = build_run_config(args=args, n_products=n_products, hidden_sizes=hidden_sizes)
    with (args.outdir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
        f.write("\n")

    best_val = float("inf")
    best_epoch = -1
    best_path = args.outdir / "best.pt"
    last_path = args.outdir / "last.pt"
    history: list[TrainResult] = []
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_nll = train_one_epoch(model, train_loader, optimizer, device)
        val_nll, val_acc = evaluate(model, val_loader, device)
        history.append(
            TrainResult(
                epoch=epoch,
                train_nll=train_nll,
                val_nll=val_nll,
                val_accuracy=val_acc,
            )
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model": args.model,
                "n_products": n_products,
                "hidden_sizes": hidden_sizes,
                "num_blocks": args.num_blocks,
                "dataset_path": str(args.data),
                "seed": args.seed,
            },
            last_path,
        )

        if val_nll < best_val:
            best_val = val_nll
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model": args.model,
                    "n_products": n_products,
                    "hidden_sizes": hidden_sizes,
                    "num_blocks": args.num_blocks,
                    "dataset_path": str(args.data),
                    "seed": args.seed,
                },
                best_path,
            )
        else:
            bad_epochs += 1

        if args.patience > 0 and bad_epochs >= args.patience:
            break

    summary = {
        "best_epoch": best_epoch,
        "best_val_nll": best_val,
        "device": str(device),
        "history": [asdict(row) for row in history],
    }
    with (args.outdir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
