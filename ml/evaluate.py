"""Evaluate the trained model on the held-out test set.

- Picks the operating threshold from the **validation** set as the smallest
  decision threshold whose specificity is at least `--min-specificity`
  (default 0.80) — favouring sensitivity for a screening task.
- Reports Accuracy, Sensitivity (Recall on malignant), Specificity, ROC AUC,
  PR AUC, and the confusion matrix on the held-out **test** set at that
  threshold.
- Persists metrics.json + roc.png to `reports/`.
- Writes the chosen threshold back into the checkpoint so the API can pick it
  up automatically.

Usage:
    uv run python evaluate.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader  # noqa: E402
from torchvision.models import efficientnet_b0  # noqa: E402
from tqdm import tqdm  # noqa: E402

from dataset import MammogramDataset  # noqa: E402

ROOT = Path(__file__).parent
SPLITS = ROOT / "splits"
IMAGE_ROOT = ROOT / "data" / "cbis-ddsm-png"
MODELS_DIR = ROOT.parent / "models"
REPORTS_DIR = ROOT.parent / "reports"
DEFAULT_MODEL = MODELS_DIR / "mammo-v1.pt"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint: Path, device: torch.device) -> tuple[nn.Module, dict]:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    model.load_state_dict(state["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    return model, state


def gather_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="predict", leave=False):
            logits = model(x.to(device)).squeeze(1)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y.numpy().astype(int))
    return np.concatenate(probs), np.concatenate(labels)


def threshold_for_min_sensitivity(
    val_labels: np.ndarray, val_probs: np.ndarray, min_sensitivity: float
) -> float:
    """Highest threshold that still achieves ≥ min_sensitivity on val.

    Screening philosophy: missed cancers (false negatives) carry far more
    weight than false positives (a recall biopsy is recoverable; a missed
    invasive cancer is not). We sweep the val ROC and pick the largest
    threshold (i.e., most conservative) that still hits the sensitivity floor.
    """
    fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
    candidates = [
        (thresh, tpr_i, 1.0 - fpr_i)
        for thresh, tpr_i, fpr_i in zip(thresholds, tpr, fpr, strict=True)
        if tpr_i >= min_sensitivity
    ]
    if not candidates:
        j = tpr - fpr
        return float(thresholds[int(np.argmax(j))])
    # Among candidates, maximise specificity (equivalently, the highest threshold).
    best = max(candidates, key=lambda t: (t[2], t[0]))
    return float(best[0])


def threshold_for_min_specificity(
    val_labels: np.ndarray, val_probs: np.ndarray, min_specificity: float
) -> float:
    """Lowest threshold that still achieves ≥ min_specificity on val (diagnostic op-point)."""
    fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
    candidates = [
        (thresh, tpr_i, 1.0 - fpr_i)
        for thresh, tpr_i, fpr_i in zip(thresholds, tpr, fpr, strict=True)
        if 1.0 - fpr_i >= min_specificity
    ]
    if not candidates:
        j = tpr - fpr
        return float(thresholds[int(np.argmax(j))])
    best = max(candidates, key=lambda t: (t[1], -t[0]))
    return float(best[0])


def threshold_youden(val_labels: np.ndarray, val_probs: np.ndarray) -> float:
    """Maximises Youden's J = sensitivity + specificity - 1 (balanced op-point)."""
    fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
    return float(thresholds[int(np.argmax(tpr - fpr))])


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "pr_auc": float(average_precision_score(labels, probs)),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n": int(len(labels)),
        "prevalence_malignant": float(labels.mean()),
    }


def plot_roc(labels: np.ndarray, probs: np.ndarray, auc: float, out: Path) -> None:
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC — CBIS-DDSM test set")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--min-sensitivity", type=float, default=0.90,
                        help="Screening operating point: minimum sensitivity on val")
    parser.add_argument("--min-specificity", type=float, default=0.80,
                        help="Diagnostic operating point: minimum specificity on val")
    parser.add_argument("--out-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args(argv)

    device = pick_device()
    print(f"Device: {device}")
    model, state = load_model(args.model, device)

    val_ds = MammogramDataset(SPLITS / "val.csv", IMAGE_ROOT, train=False)
    test_ds = MammogramDataset(SPLITS / "test.csv", IMAGE_ROOT, train=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers)

    print(f"Val: {len(val_ds)}  Test: {len(test_ds)}")

    val_probs, val_labels = gather_probs(model, val_loader, device)
    test_probs, test_labels = gather_probs(model, test_loader, device)

    # We report three operating points so the report can be honest about
    # the precision/recall trade-space. The API uses the SCREENING point.
    t_screening = threshold_for_min_sensitivity(val_labels, val_probs, args.min_sensitivity)
    t_diagnostic = threshold_for_min_specificity(val_labels, val_probs, args.min_specificity)
    t_balanced = threshold_youden(val_labels, val_probs)
    threshold = t_screening

    print(f"Operating thresholds (chosen on val):")
    print(f"  screening   (sens ≥ {args.min_sensitivity:.2f}): {t_screening:.4f}")
    print(f"  diagnostic  (spec ≥ {args.min_specificity:.2f}): {t_diagnostic:.4f}")
    print(f"  balanced    (Youden's J):                  {t_balanced:.4f}")
    print(f"API will use: screening = {threshold:.4f}")

    operating_points = {
        "screening": {
            "threshold": t_screening,
            "rationale": (
                f"Lowest false-negative rate while keeping val sensitivity ≥ "
                f"{args.min_sensitivity:.2f}. Used by the API."
            ),
            "test": compute_metrics(test_labels, test_probs, t_screening),
            "val": compute_metrics(val_labels, val_probs, t_screening),
        },
        "diagnostic": {
            "threshold": t_diagnostic,
            "rationale": (
                f"Highest specificity while keeping val specificity ≥ "
                f"{args.min_specificity:.2f}. Useful for follow-up reads."
            ),
            "test": compute_metrics(test_labels, test_probs, t_diagnostic),
            "val": compute_metrics(val_labels, val_probs, t_diagnostic),
        },
        "balanced": {
            "threshold": t_balanced,
            "rationale": "Youden's J — maximises sensitivity + specificity − 1 on val.",
            "test": compute_metrics(test_labels, test_probs, t_balanced),
            "val": compute_metrics(val_labels, val_probs, t_balanced),
        },
        "default_0.5": {
            "threshold": 0.5,
            "rationale": "Sigmoid default; reported for transparency only.",
            "test": compute_metrics(test_labels, test_probs, 0.5),
            "val": compute_metrics(val_labels, val_probs, 0.5),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_roc(test_labels, test_probs, operating_points["screening"]["test"]["roc_auc"],
             args.out_dir / "roc.png")

    bundle = {
        "model_version": state.get("model_version"),
        "arch": state.get("arch"),
        "best_val_auc_during_training": state.get("best_val_auc"),
        "api_operating_point": "screening",
        "api_threshold": threshold,
        "operating_points": operating_points,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(bundle, indent=2))
    print(json.dumps(bundle, indent=2))

    # Update checkpoint with the chosen threshold so the API picks it up.
    state["threshold"] = threshold
    torch.save(state, args.model)
    print(f"\nUpdated {args.model} with operating threshold {threshold:.4f}")
    print(f"Wrote {args.out_dir / 'metrics.json'} and {args.out_dir / 'roc.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
