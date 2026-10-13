"""
PyTorch Dataset and DataLoader for Vehicle Speed Estimation.

Loads optical flow images and returns (flow_image, speed_label) pairs.
Splits the sequential data into train and validation sets to avoid
temporal leakage.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path


def load_labels(label_path: str) -> np.ndarray:
    """Load speed labels from a text file."""
    with open(label_path, "r") as f:
        lines = f.read().strip().splitlines()
    return np.array([float(x) for x in lines], dtype=np.float32)


class SpeedDataset(Dataset):
    """
    Dataset for vehicle speed estimation from optical flow images.

    Notes on indexing:
    - Flow image i represents motion between frame i and frame i+1.
    - The label for flow image i is labels[i+1] (the speed at frame i+1).
    - Frame 0 has no prior frame, so flow indices start at 0 and use labels 1..N.
    """

    def __init__(
        self,
        flow_dir: str,
        label_path: str = None,  # None for unlabeled test data
        transform=None,
    ):
        self.flow_dir = Path(flow_dir)
        self.image_paths = sorted(self.flow_dir.glob("flow_*.png"))
        self.transform = transform

        if label_path is not None:
            self.labels = load_labels(label_path)
            # Flow image i corresponds to label i+1 (end-frame of the interval)
            # We have len(image_paths) flow images, need labels[1 : len+1]
            num_flows = len(self.image_paths)
            self.labels = self.labels[1 : num_flows + 1]
            assert len(self.labels) == len(self.image_paths), (
                f"Mismatch: {len(self.image_paths)} flow images but "
                f"{len(self.labels)} available labels."
            )
            self.has_labels = True
        else:
            self.labels = None
            self.has_labels = False

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            # Default: ToTensor normalizes to [0, 1]
            image = transforms.ToTensor()(image)

        if self.has_labels:
            speed = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, speed
        else:
            return image


def build_dataloaders(
    flow_dir: str,
    label_path: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 4,
    augment_train: bool = True,
):
    """
    Build train and validation DataLoaders with temporal split.

    Args:
        flow_dir: Directory containing flow images.
        label_path: Path to train.txt labels.
        batch_size: Batch size for both loaders.
        val_split: Fraction of data to use for validation (temporal end).
        num_workers: Number of DataLoader workers.
        augment_train: If True, apply random augmentations to training data.

    Returns:
        train_loader, val_loader
    """
    all_paths = sorted(Path(flow_dir).glob("flow_*.png"))
    num_total = len(all_paths)
    num_val = int(num_total * val_split)
    num_train = num_total - num_val

    # Temporal split: first 80% train, last 20% val
    train_paths = all_paths[:num_train]
    val_paths = all_paths[num_train:]

    # Split labels accordingly
    all_labels = load_labels(label_path)
    train_labels = all_labels[1 : num_train + 1]
    val_labels = all_labels[num_train + 1 : num_total + 1]

    # Build transforms
    base_transform = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]

    if augment_train:
        train_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            *base_transform,
        ])
    else:
        train_transform = transforms.Compose(base_transform)

    val_transform = transforms.Compose(base_transform)

    train_dataset = SpeedDataset.__new__(SpeedDataset)
    train_dataset.flow_dir = Path(flow_dir)
    train_dataset.image_paths = train_paths
    train_dataset.transform = train_transform
    train_dataset.labels = train_labels
    train_dataset.has_labels = True

    val_dataset = SpeedDataset.__new__(SpeedDataset)
    val_dataset.flow_dir = Path(flow_dir)
    val_dataset.image_paths = val_paths
    val_dataset.transform = val_transform
    val_dataset.labels = val_labels
    val_dataset.has_labels = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    return train_loader, val_loader


def build_test_dataloader(
    flow_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
):
    """Build a DataLoader for unlabeled test data."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = SpeedDataset(flow_dir=flow_dir, label_path=None, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader
