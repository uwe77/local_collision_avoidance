from typing import Dict
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

class LaserEncoder(nn.Module):
    """
    Encodes a 1D laser scan into a flattened feature vector using two Conv1d layers.
    Input: (batch_size, scan_length)
    Output: (batch_size, out_dim)
    """
    def __init__(self, scan_length: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # Determine output dimension dynamically
        with th.no_grad():
            dummy = th.zeros(1, 1, scan_length)
            out = self.conv(dummy)
            self.out_dim = out.shape[1] * out.shape[2]

    def forward(self, x: th.Tensor) -> th.Tensor:
        # x: (batch_size, scan_length)
        x = x.unsqueeze(1)         # -> (batch_size, 1, scan_length)
        x = self.conv(x)           # -> (batch_size, 32, scan_length)
        return x.reshape(x.size(0), -1)  # Flatten

class USVFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for USV environment.
    Processes 'laser' with Conv1d, flattens 'track' and 'vel',
    then concatenates and applies an MLP to obtain features_dim.
    """
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 512,
    ):
        super().__init__(observation_space, features_dim)

        # Create encoders
        self.extractors = nn.ModuleDict()
        total_dim = 0

        # Laser encoder
        laser_space = observation_space.spaces["laser"]
        scan_len = laser_space.shape[0]
        self.extractors["laser"] = LaserEncoder(scan_len)
        total_dim += self.extractors["laser"].out_dim

        # Track and vel flatteners
        for key in ["track", "vel"]:
            subspace = observation_space.spaces[key]
            self.extractors[key] = nn.Flatten()
            total_dim += subspace.shape[0]

        # Final MLP to desired features_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, features_dim),
            nn.ELU(),
            nn.Linear(features_dim, features_dim),
            nn.ELU(),
        )

        self._features_dim = features_dim

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        # Encode each part
        laser_feat = self.extractors["laser"](observations["laser"])
        track_feat = self.extractors["track"](observations["track"])
        vel_feat = self.extractors["vel"](observations["vel"])

        # Concatenate
        concat = th.cat([laser_feat, track_feat, vel_feat], dim=1)
        # MLP
        return self.mlp(concat)
