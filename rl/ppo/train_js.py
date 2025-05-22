import torch as th
import torch.nn as nn
from gymnasium import spaces
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, StopTrainingOnRewardThreshold
from stable_baselines3 import PPO

from js_feature_extractor import JsFeatureExtractor
from ppo_callback import OverwriteCheckpointCallback
from datetime import date


# Parallel environments
vec_env = make_vec_env("gymnasium_vrx:js-collision-v0", n_envs=1)

policy_kwargs = dict(
    features_extractor_class=JsFeatureExtractor,
    net_arch=(dict(pi=[512, 256, 256], vf=[512, 256, 256])),
    activation_fn = th.nn.ELU,
)

today = date.today()
checkpoint_callback = OverwriteCheckpointCallback(
    save_freq=50000,
    save_path="./logs/",
    name_prefix="ppo_js_nav",
    save_replay_buffer=True,
    save_vecnormalize=True,
    overwrite=True,  # ✅ This enables overwrite mode
)

model = PPO("MultiInputPolicy", vec_env, 
            verbose=1, 
            policy_kwargs=policy_kwargs, 
            batch_size=128,
            n_steps=4096,
            n_epochs=1, 
            ent_coef=0.01,
            gae_lambda=0.95,
            learning_rate=5e-6,
            tensorboard_log='tb_ppo'
            )
            
model.learn(total_timesteps=20_000_000, tb_log_name='tb_ppo', callback=checkpoint_callback)
model.save("ppo_js_nav")