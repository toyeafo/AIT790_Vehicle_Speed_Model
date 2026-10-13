"""
Utility functions for training and post-processing.

Includes:
- Speed thresholding (stop rule)
- Rolling average smoothing
- Metrics (MAE, RMSE, R^2)
"""

import numpy as np
import torch
import torch.nn as nn


class RollingAverageFilter:
    """
    Rolling average smoothing filter for live speed display.
    Maintains a buffer of the last N predictions.
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.buffer = []

    def update(self, value: float) -> float:
        self.buffer.append(value)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        return float(np.mean(self.buffer))

    def reset(self):
        self.buffer = []


def apply_speed_threshold(speed: float, threshold: float = 2.0) -> float:
    """
    Post-processing rule: if predicted speed <= threshold, return 0.
    This handles the 'stopped vehicle' case.
    """
    return 0.0 if speed <= threshold else speed


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression metrics: MSE, MAE, RMSE.

    Args:
        y_true: Ground truth speeds.
        y_pred: Predicted speeds.

    Returns:
        Dictionary with metric values.
    """
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))
    return {"mse": mse, "mae": mae, "rmse": rmse}


class EarlyStopping:
    """Stop training if validation loss does not improve after N epochs."""

    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False
