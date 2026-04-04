#!/usr/bin/env python3
"""Train a discrete Bernoulli diffusion model on assortment states."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, random_split

from bernoulli_diffusion import BernoulliDiffusion
from diffusion_dataset import AssortmentStateDataset


@dataclass
class TrainResult:
    epoch: int
    train_loss: float
    val_loss: float
    val_bit_accuracy: float


def parse_sizes(spec: str | None, default: Sequence[int]) -> list[int]:
    if spec is None or not spec.strip():
        return list(default)
    return [int(x) for x in spec.split(",") if x.strip()]


@torch.no_grad()
def evaluate(
    model: BernoulliDiffusion,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_bits = 0
    total_correct = 0.0

    for batch in loader:
        x0 = batch["S"].to(device)
        t = torch.randint(1, model.num_steps + 1, (x0.shape[0],), device=device)
        xt = model.corrupt(x0, t)
        logits = model.forward(xt, t)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, x0, reduction="sum")
        total_loss += float(loss.item())
        preds = (torch.sigmoid(logits) >= 0.5).float()
        total_correct += float((preds == x0).float().sum().item())
        total_bits += int(x0.numel())

    if total_bits == 0:
        raise ValueError("Loader produced no examples.")

    return total_loss / total_bits, total_correct / total_bits


def train_one_epoch(
    model: BernoulliDiffusion,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_bits = 0

    for batch in loader:
        x0 = batch["S"].to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = model.denoising_loss(x0, reduction="mean")
        loss.backward()
        optimizer.step()

        batch_size = int(x0.shape[0])
        total_loss += float(loss.item()) * batch_size * model.n_products
        total_bits += batch_size * model.n_products

    if total_bits == 0:
        raise ValueError("Loader produced no examples.")

    return total_loss / total_bits


def build_run_config(
    *,
    args: argparse.Namespace,
    n_products: int,
    hidden_sizes: Sequence[int],
) -> dict:
    architecture = {
        "name": "bernoulli_diffusion",
        "n_products": n_products,
        "num_steps": args.num_steps,
        "beta_start": args.beta_start,
        "beta_end": args.beta_end,
        "hidden_sizes": list(hidden_sizes),
        "time_emb_dim": args.time_emb_dim,
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


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Bernoulli diffusion model on assortment states."
    )
    parser.add_argument("--data", type=Path, required=True, help="Path to JSON dataset.")
    parser.add_argument(
        "--hidden-sizes",
        type=str,
        default="128,128",
        help="Comma-separated MLP hidden sizes. Default: '128,128'.",
    )
    parser.add_argument("--num-steps", type=int, default=50, help="Diffusion timesteps.")
    parser.add_argument("--beta-start", type=float, default=1e-4, help="Initial beta.")
    parser.add_argument("--beta-end", type=float, default=0.1, help="Final beta.")
    parser.add_argument("--time-emb-dim", type=int, default=32, help="Time embedding size.")
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
        default=Path("outputs/bernoulli_diffusion"),
        help="Directory for checkpoints and metrics.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early-stopping patience on validation loss.",
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
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive.")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = AssortmentStateDataset(args.data)
    n_products = dataset.n_products
    hidden_sizes = parse_sizes(args.hidden_sizes, default=(128, 128))

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

    model = BernoulliDiffusion(
        n_products=n_products,
        num_steps=args.num_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        hidden_sizes=hidden_sizes,
        time_emb_dim=args.time_emb_dim,
    ).to(device)
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
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        history.append(
            TrainResult(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_bit_accuracy=val_acc,
            )
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model": "bernoulli_diffusion",
                "n_products": n_products,
                "num_steps": args.num_steps,
                "beta_start": args.beta_start,
                "beta_end": args.beta_end,
                "hidden_sizes": hidden_sizes,
                "time_emb_dim": args.time_emb_dim,
                "dataset_path": str(args.data),
                "seed": args.seed,
            },
            last_path,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model": "bernoulli_diffusion",
                    "n_products": n_products,
                    "num_steps": args.num_steps,
                    "beta_start": args.beta_start,
                    "beta_end": args.beta_end,
                    "hidden_sizes": hidden_sizes,
                    "time_emb_dim": args.time_emb_dim,
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
        "best_val_loss": best_val,
        "device": str(device),
        "history": [asdict(row) for row in history],
    }
    with (args.outdir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

