"""
Final evaluation script for the Vehicle Speed Estimation model.

Loads the best checkpoint and evaluates on the full validation set,
producing detailed metrics, per-speed-bin analysis, and visualizations.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

from model import get_model
from dataset import build_dataloaders
from utils import compute_metrics


def evaluate_model(model, val_loader, device, output_dir="outputs"):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, speeds in val_loader:
            images = images.to(device)
            outputs = model(images)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(speeds.numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    # Overall metrics
    metrics = compute_metrics(y_true, y_pred)
    print("=" * 50)
    print("FINAL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Validation samples: {len(y_true):,}")
    print(f"  MSE:   {metrics['mse']:.4f}")
    print(f"  MAE:   {metrics['mae']:.4f} mph")
    print(f"  RMSE:  {metrics['rmse']:.4f} mph")
    print("=" * 50)

    # Per-speed-bin analysis
    bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30)]
    print("\nPer-Speed-Range Performance:")
    print(f"{'Range':<12} {'Count':>8} {'MAE':>8} {'RMSE':>8} {'Max Error':>10}")
    print("-" * 50)
    for lo, hi in bins:
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() == 0:
            continue
        m = compute_metrics(y_true[mask], y_pred[mask])
        max_err = float(np.max(np.abs(y_true[mask] - y_pred[mask])))
        print(f"{lo}-{hi} mph    {mask.sum():>8} {m['mae']:>8.3f} {m['rmse']:>8.3f} {max_err:>10.3f}")

    # Save metrics JSON
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {output_dir / 'final_metrics.json'}")

    # Plot: Scatter plot of predicted vs actual
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Scatter
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.3, s=8, c="tab:blue")
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2, label="Perfect prediction")
    ax.set_xlabel("Actual Speed (mph)")
    ax.set_ylabel("Predicted Speed (mph)")
    ax.set_title(f"Predicted vs Actual (MAE={metrics['mae']:.2f} mph)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error histogram
    ax = axes[0, 1]
    errors = y_pred - y_true
    ax.hist(errors, bins=100, color="tab:orange", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", lw=2)
    ax.set_xlabel("Prediction Error (mph)")
    ax.set_ylabel("Count")
    ax.set_title(f"Error Distribution (std={errors.std():.2f})")
    ax.grid(True, alpha=0.3)

    # Absolute error vs speed
    ax = axes[1, 0]
    abs_err = np.abs(errors)
    ax.scatter(y_true, abs_err, alpha=0.3, s=8, c="tab:green")
    ax.axhline(metrics["mae"], color="red", linestyle="--", lw=2, label=f"Mean MAE = {metrics['mae']:.2f}")
    ax.set_xlabel("Actual Speed (mph)")
    ax.set_ylabel("Absolute Error (mph)")
    ax.set_title("Error vs. Actual Speed")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cumulative error distribution
    ax = axes[1, 1]
    sorted_err = np.sort(abs_err)
    cum_pct = np.arange(1, len(sorted_err) + 1) / len(sorted_err) * 100
    ax.plot(sorted_err, cum_pct, lw=2, color="tab:purple")
    ax.axvline(metrics["mae"], color="red", linestyle="--", lw=2, label=f"MAE = {metrics['mae']:.2f}")
    ax.axhline(80, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Absolute Error (mph)")
    ax.set_ylabel("Cumulative % of Predictions")
    ax.set_title("Cumulative Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(15, sorted_err.max()))

    plt.tight_layout()
    plt.savefig(output_dir / "evaluation_plots.png", dpi=150)
    plt.close()
    print(f"Saved evaluation plots to {output_dir / 'evaluation_plots.png'}")

    return metrics


def main():
    print("Loading model and data for final evaluation...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    _, val_loader = build_dataloaders(
        flow_dir="data/flow",
        label_path="data/train.txt",
        batch_size=32,
        val_split=0.2,
        num_workers=2,
        augment_train=False,
    )

    model, _ = get_model(device=device)
    checkpoint = torch.load("outputs/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint)

    evaluate_model(model, val_loader, device)


if __name__ == "__main__":
    main()
