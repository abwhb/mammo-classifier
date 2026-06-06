"""Train EfficientNet-B0 binary classifier on CBIS-DDSM PNG mirror.

Auto-detects compute device: CUDA → MPS → CPU. Optimised for a single Apple
Silicon machine; falls back gracefully to CPU.

Outputs:
    models/mammo-v1.pt  — best checkpoint by val AUC

Usage:
    uv run python train.py
    uv run python train.py --epochs 10 --batch-size 32 --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from tqdm import tqdm

from dataset import MammogramDataset

ROOT = Path(__file__).parent
SPLITS = ROOT / "splits"
IMAGE_ROOT = ROOT / "data" / "cbis-ddsm-png"
MODELS_DIR = ROOT.parent / "models"
MODEL_VERSION = "mammo-v1"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model() -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    """Returns (mean_loss, auc). When optimizer is None, runs in eval mode."""
    is_train = optimizer is not None
    model.train(mode=is_train)
    losses: list[float] = []
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    autograd_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with autograd_ctx:
        for x, y in tqdm(loader, leave=False, desc="train" if is_train else "eval "):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            if is_train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            all_logits.append(logits.detach().cpu())
            all_labels.append(y.detach().cpu())

    logits_t = torch.cat(all_logits)
    labels_t = torch.cat(all_labels)
    probs = torch.sigmoid(logits_t).numpy()
    labels = labels_t.numpy().astype(int)
    auc = float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else float("nan")
    return sum(losses) / max(len(losses), 1), auc


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=3, help="early-stop patience on val AUC")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=MODELS_DIR / f"{MODEL_VERSION}.pt")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"Device: {device}")

    train_ds = MammogramDataset(SPLITS / "train.csv", IMAGE_ROOT, train=True)
    val_ds = MammogramDataset(SPLITS / "val.csv", IMAGE_ROOT, train=False)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # On MPS, persistent_workers helps; on CPU we keep it simple.
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )

    model = build_model().to(device)
    pos_weight = train_ds.pos_weight().to(device)
    print(f"Class imbalance pos_weight: {pos_weight.item():.3f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc = -1.0
    best_epoch = -1
    epochs_since_best = 0
    history: list[dict[str, float]] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_auc = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_auc = run_epoch(model, val_loader, device, criterion, optimizer=None)
        scheduler.step()
        dt = time.perf_counter() - t0
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_auc": train_auc,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "lr": float(scheduler.get_last_lr()[0]),
                "seconds": dt,
            }
        )
        print(
            f"epoch {epoch:02d}/{args.epochs}  "
            f"train_loss={train_loss:.4f} train_auc={train_auc:.4f}  "
            f"val_loss={val_loss:.4f} val_auc={val_auc:.4f}  "
            f"({dt:.1f}s)"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            epochs_since_best = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_version": MODEL_VERSION,
                    "arch": "efficientnet_b0",
                    "threshold": 0.5,  # placeholder; chosen in evaluate.py
                    "best_val_auc": best_auc,
                    "epoch": epoch,
                    "history": history,
                },
                args.out,
            )
            print(f"  ↑ best val_auc={best_auc:.4f}  → saved {args.out}")
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(
                    f"Early stop: no val_auc improvement for {args.patience} epochs "
                    f"(best={best_auc:.4f} @ epoch {best_epoch})"
                )
                break

    (ROOT.parent / "reports" / "train_history.json").write_text(
        json.dumps({"history": history, "best_val_auc": best_auc, "best_epoch": best_epoch}, indent=2)
    )
    print(f"\nDone. Best val_auc={best_auc:.4f} @ epoch {best_epoch}. Checkpoint: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
