import torch as th
import torch.nn as nn
from gymnasium import spaces
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from ppo_callback import OverwriteCheckpointCallback

from datetime import date


# Parallel environments
vec_env = make_vec_env("gymnasium_vrx:js-collision-v0", n_envs=1)


today = date.today()
checkpoint_callback = OverwriteCheckpointCallback(
    save_freq=50000,
    save_path="./logs/",
    name_prefix="ppo_js_nav",
    save_replay_buffer=True,
    save_vecnormalize=True,
    overwrite=True,  # ✅ This enables overwrite mode
)
model = PPO.load("ppo_js_nav")
model.entropy_coef = 0.01
model.learning_rate = 1e-6

model.set_env(vec_env)
model.learn(total_timesteps=20_000_000, tb_log_name='tb_ppo', callback=checkpoint_callback)
model.save("ppo_js_nav")