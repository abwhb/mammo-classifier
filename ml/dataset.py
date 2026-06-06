"""PyTorch Dataset for CBIS-DDSM PNGs read from a split CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


def build_transforms(train: bool) -> transforms.Compose:
    if train:
        # Mammogram-appropriate augmentation:
        #   - Horizontal flip is fine (mirror image of the contralateral breast)
        #   - Small rotations (chest wall orientation roughly invariant)
        #   - Mild crop to simulate framing variability
        #   - NO vertical flip (anatomy has consistent top/bottom in standard views)
        return transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(int(IMAGE_SIZE * 1.15)),
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class MammogramDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, csv_path: Path, image_root: Path, train: bool) -> None:
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.tf = build_transforms(train=train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img = Image.open(self.image_root / row["path"]).convert("L")
        x = self.tf(img)
        y = torch.tensor(float(row["label"]), dtype=torch.float32)
        return x, y

    def pos_weight(self) -> torch.Tensor:
        """Positive-class weight for BCEWithLogitsLoss to combat class imbalance."""
        pos = int((self.df["label"] == 1).sum())
        neg = int((self.df["label"] == 0).sum())
        if pos == 0:
            return torch.tensor(1.0)
        return torch.tensor(max(neg / pos, 1.0), dtype=torch.float32)
