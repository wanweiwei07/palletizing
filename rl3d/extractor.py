"""Feature extractor: small CNN for heightmap + MLP for box features."""
from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PalletExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim)
        h_shape = observation_space.spaces["heightmap"].shape  # (1, GX, GY)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *h_shape)
            cnn_out = self.cnn(dummy).shape[1]

        n = observation_space.spaces["remaining"].shape[0]
        self.box_mlp = nn.Sequential(
            nn.Linear(n + n * 3, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(cnn_out + 128, features_dim), nn.ReLU(),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.cnn(obs["heightmap"])
        remaining = obs["remaining"]
        dims = obs["box_dims"].flatten(start_dim=1)
        b = self.box_mlp(torch.cat([remaining, dims], dim=1))
        return self.fusion(torch.cat([h, b], dim=1))
