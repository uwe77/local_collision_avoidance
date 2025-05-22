from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 1):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        extractors = {}
        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == 'laser':
                # We will just downsample one channel of the laser by 4x241 and flatten.
                # Assume the laser is single-channel (subspace.shape[0] == 0)
                extractors[key] = nn.Sequential(
                        nn.Conv1d(subspace.shape[0], 32, kernel_size=3, stride=1),
                        nn.ReLU(),
                        nn.Conv1d(32, 32, kernel_size=3, stride=1),
                        nn.ReLU(),
                        nn.Flatten(),
                    )
                # print("laser", subspace.shape)
                total_concat_size += 7584
            elif key == 'track' or key == 'vel' or key == 'action':
                # Run through nothing
                extractors[key] = nn.Sequential()
                total_concat_size += subspace.shape[0]
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size
        # Update the features dim manually
        # self._features_dim = total_concat_size
        # self.mlp_network = nn.Sequential(
        #     nn.Linear(total_concat_size, 512),
        #     nn.ELU(),
        #     nn.Linear(512, 256),
        #     nn.ELU(),
        #     nn.Linear(256, 256),
        #     nn.ELU(),
        # )
        # self._features_dim = 256



    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        features = th.cat(encoded_tensor_list, dim=1)
        return features
        # return self.mlp_network(features)