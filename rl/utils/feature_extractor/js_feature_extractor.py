from typing import Dict
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces


class JsFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        self.extractors = nn.ModuleDict()
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if key == 'laser':
                self.extractors[key] = nn.Sequential(
                    nn.Conv1d(subspace.shape[0], 32, kernel_size=3, stride=1),
                    nn.ReLU(),
                    nn.Conv1d(32, 32, kernel_size=3, stride=1),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                # Dynamically compute flattened size
                with th.no_grad():
                    dummy_input = th.zeros(1, subspace.shape[0], subspace.shape[1])  # (B, C, L)
                    laser_out = self.extractors[key](dummy_input)
                    total_concat_size += laser_out.shape[1]
                # self.extractors[key] = nn.Flatten()
                # total_concat_size += subspace.shape[0] * subspace.shape[1]

            elif key == 'track':
                # Flatten (10, 3) → 30
                self.extractors[key] = nn.Flatten()
                total_concat_size += subspace.shape[0] * subspace.shape[1]

            elif key == 'vel':
                # Flatten (10, 1) → 10
                self.extractors[key] = nn.Flatten()
                total_concat_size += subspace.shape[0] * subspace.shape[1]

        # MLP after concatenation
        self.mlp_network = nn.Sequential(
            # nn.Linear(total_concat_size, 512),
            # nn.ELU(),
            # nn.Linear(512, 256),
            # nn.ELU(),
            # nn.Linear(256, 256),
            # nn.ELU(),
            nn.Flatten(),
        )

        self._features_dim = total_concat_size

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            x = observations[key]
            # Laser already shaped (B, 4, 241), no need to unsqueeze
            encoded_tensor_list.append(extractor(x))

        concat_features = th.cat(encoded_tensor_list, dim=1)
        return self.mlp_network(concat_features)
