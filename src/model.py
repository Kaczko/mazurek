"""Classic LeNet-5 convolutional network (LeCun et al., 1998).

Architecture (input 1x32x32):

    C1: conv 6  @ 5x5   -> 6  x 28 x 28
    S2: avg-pool 2x2    -> 6  x 14 x 14
    C3: conv 16 @ 5x5   -> 16 x 10 x 10
    S4: avg-pool 2x2    -> 16 x  5 x  5
    C5: conv 120 @ 5x5  -> 120 x 1 x 1
    F6: fully connected -> 84
    out: fully connected -> num_classes
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),     # C1 -> 6x28x28
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # S2 -> 6x14x14
            nn.Conv2d(6, 16, kernel_size=5),    # C3 -> 16x10x10
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # S4 -> 16x5x5
            nn.Conv2d(16, 120, kernel_size=5),  # C5 -> 120x1x1
            nn.Tanh(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(120, 84),                 # F6
            nn.Tanh(),
            nn.Linear(84, num_classes),         # output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
