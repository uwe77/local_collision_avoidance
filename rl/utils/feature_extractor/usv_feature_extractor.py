from typing import Dict
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class LaserEncoder(nn.Module):
    """
    Encodes a 1D laser scan into a 128-D feature vector using Conv+Pool.
    Input: (batch_size, scan_length)
    Output: (batch_size, 128)
    """
    def __init__(self, scan_length: int, features_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),                # half length
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(16)        # to length 16
        )
        conv_out_dim = 64 * 16
        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.GELU(),
            nn.Dropout(p=0.1)
        )
        self.out_dim = features_dim

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = x.unsqueeze(1)                 # (B,1,L)
        x = self.conv(x)                   # (B,64,16)
        x = x.view(x.size(0), -1)          # (B,64*16)
        return self.fc(x)                  # (B,128)


class TrackEncoder(nn.Module):
    """
    Embeds track state (dist, angle) into an 8-D feature.
    """
    def __init__(self, input_dim: int, embed_dim: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, embed_dim),
            nn.GELU()
        )
        self.out_dim = embed_dim

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.fc(x)


class VelEncoder(nn.Module):
    """
    Embeds velocity vector into an 8-D feature.
    """
    def __init__(self, input_dim: int, embed_dim: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, embed_dim),
            nn.GELU()
        )
        self.out_dim = embed_dim

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.fc(x)


class ResidualFusion(nn.Module):
    """
    Residual MLP block: input -> hidden -> hidden + skip -> output.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        # project skip if needed
        self.residual_proj = nn.Identity() if input_dim == hidden_dim else nn.Linear(input_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(p=0.1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        h = self.act(self.norm1(self.fc1(x)))
        res = self.residual_proj(x)
        h2 = self.act(self.norm2(self.fc2(h)) + res)
        out = self.fc3(h2)
        return self.drop(out)


class USVFeatureExtractor(BaseFeaturesExtractor):
    """
    Enhanced USV feature extractor:
      - Laser: Conv+Pool → 128-D
      - Track: MLP → 8-D
      - Vel:   MLP → 8-D
      - Fusion: Residual MLP → features_dim
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # Laser encoder
        scan_len = observation_space.spaces["laser"].shape[0]
        self.laser_enc = LaserEncoder(scan_len, features_dim=128)
        # Track encoder
        track_dim = observation_space.spaces["track"].shape[0]
        self.track_enc = TrackEncoder(track_dim, embed_dim=8)
        # Velocity encoder
        vel_dim = observation_space.spaces["vel"].shape[0]
        self.vel_enc = VelEncoder(vel_dim, embed_dim=8)
        # Fusion block
        fusion_in = self.laser_enc.out_dim + self.track_enc.out_dim + self.vel_enc.out_dim
        self.res_fuse = ResidualFusion(fusion_in, hidden_dim=128, output_dim=features_dim)
        self._features_dim = features_dim

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        laser = self.laser_enc(observations["laser"])
        track = self.track_enc(observations["track"])
        vel   = self.vel_enc(observations["vel"])
        fusion = th.cat([laser, track, vel], dim=1)
        return self.res_fuse(fusion)
