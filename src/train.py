"""
Training script for the Vehicle Speed Estimation model.

Usage:
    venv/bin/python src/train.py --epochs 30 --batch-size 32 --lr 1e-3
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from model import get_model
from dataset import build_dataloaders
from utils import compute_metrics, EarlyStopping


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, speeds in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device)
        speeds = speeds.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, speeds)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        all_preds.extend(outputs.detach().cpu().numpy())
        all_targets.extend(speeds.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_metrics(np.array(all_targets), np.array(all_preds))
    return epoch_loss, metrics


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, speeds in tqdm(dataloader, desc="Val", leave=False):
            images = images.to(device)
            speeds = speeds.to(device)

            outputs = model(images)
            loss = criterion(outputs, speeds)

            running_loss += loss.item() * images.size(0)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(speeds.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_metrics(np.array(all_targets), np.array(all_preds))
    return epoch_loss, metrics, np.array(all_targets), np.array(all_preds)


def plot_history(history: dict, save_path: str):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history["train_mae"], label="Train MAE")
    axes[1].plot(epochs, history["val_mae"], label="Val MAE")
    axes[1].set_title("MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(epochs, history["train_rmse"], label="Train RMSE")
    axes[2].plot(epochs, history["val_rmse"], label="Val RMSE")
    axes[2].set_title("RMSE")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved training curves to {save_path}")


def plot_predictions(y_true, y_pred, save_path: str):
    """Plot predicted vs actual speeds for the final validation set."""
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="Actual", alpha=0.8)
    plt.plot(y_pred, label="Predicted", alpha=0.8)
    plt.xlabel("Frame Index")
    plt.ylabel("Speed")
    plt.title("Validation Set: Predicted vs Actual Speed")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved prediction plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train vehicle speed estimation model.")
    parser.add_argument("--flow-dir", type=str, default="data/flow", help="Directory with flow images.")
    parser.add_argument("--label-path", type=str, default="data/train.txt", help="Path to labels.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation fraction (temporal split).")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for checkpoints and plots.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(
        flow_dir=args.flow_dir,
        label_path=args.label_path,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        augment_train=True,
    )

    model, _ = get_model(device=device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    early_stopper = EarlyStopping(patience=args.patience)

    history = {
        "train_loss": [], "val_loss": [],
        "train_mae": [], "val_mae": [],
        "train_rmse": [], "val_rmse": [],
    }

    best_val_loss = float("inf")
    best_model_path = output_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, y_true, y_pred = validate_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mae"].append(train_metrics["mae"])
        history["val_mae"].append(val_metrics["mae"])
        history["train_rmse"].append(train_metrics["rmse"])
        history["val_rmse"].append(val_metrics["rmse"])

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"  Train — Loss: {train_loss:.4f} | MAE: {train_metrics['mae']:.3f} | RMSE: {train_metrics['rmse']:.3f}")
        print(f"  Val   — Loss: {val_loss:.4f} | MAE: {val_metrics['mae']:.3f} | RMSE: {val_metrics['rmse']:.3f} | LR: {current_lr:.1e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  💾 Saved best model (val_loss={val_loss:.4f})")

        if early_stopper(val_loss):
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    # Save history and plots
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    plot_history(history, str(output_dir / "training_curves.png"))
    plot_predictions(y_true, y_pred, str(output_dir / "val_predictions.png"))

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main()
