import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, n_actions: int, in_channels: int = 4):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = 64 * 7 * 7

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        features = self.conv(x)
        features = features.reshape(features.size(0), -1)
        return self.fc(features)
