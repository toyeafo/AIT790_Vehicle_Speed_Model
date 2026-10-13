"""
CNN Model for Vehicle Speed Estimation from Optical Flow Images.

A lightweight convolutional regression network designed to run fast
enough for real-time inference while maintaining accuracy.
"""

import torch
import torch.nn as nn


class SpeedEstimatorCNN(nn.Module):
    """
    Lightweight CNN that regresses vehicle speed from an optical flow image.
    Input: (B, 3, 224, 224) flow visualization image.
    Output: (B, 1) speed estimate.
    """

    def __init__(self, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 112x112

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 56x56

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 28x28

            # Block 4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 14x14

            # Block 5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),           # 256
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),     # Raw speed output
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.regressor(x)
        return x.squeeze(-1)  # (B,)


def get_model(device: str = "cuda") -> SpeedEstimatorCNN:
    """Factory: instantiate model and move to device."""
    model = SpeedEstimatorCNN()
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    else:
        device = "cpu"
        model = model.cpu()
    return model, device
