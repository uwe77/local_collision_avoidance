import add_path
import torch as th
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

from utils.callback.ppo_save_same_model_cb import PPOSaveSameModelCB
from utils.feature_extractor.usv_feature_extractor import USVFeatureExtractor
from datetime import date


# Parallel environments
max_steps = 1024
vec_env = make_vec_env(
    "gymnasium_usv:usv-local-collision-avoidance-v0", 
    env_kwargs={
        "render_mode": "none",
        "usv_name": "js",
        "obstacle_name": "redball",
        "obstacle_numbers": 4,
        "obstacle_max_speed": 10.0,
        "reset_range": 400.0,
        "reset_weight": 0.5,
        "max_steps": max_steps,},
    n_envs=1
    )

policy_kwargs = dict(
    features_extractor_class=USVFeatureExtractor,
    # net_arch=(dict(pi=[512, 256, 256], vf=[512, 256, 256])),
    activation_fn = th.nn.ELU,
)

today = date.today()
checkpoint_callback = PPOSaveSameModelCB(
    save_freq=50000,
    save_path="./logs/",
    name_prefix="ppo_usv",
    save_replay_buffer=True,
    save_vecnormalize=True,
    overwrite=True,  # ✅ This enables overwrite mode
)

model = PPO("MultiInputPolicy", vec_env, 
            verbose=1, 
            policy_kwargs=policy_kwargs, 
            batch_size=64,
            ent_coef=0.01,
            gae_lambda=0.98,
            gamma=0.999,
            n_epochs=4, 
            n_steps=max_steps,
            learning_rate=3e-4,
            tensorboard_log='tb_ppo'
            )
            
model.learn(total_timesteps=20_000_000, tb_log_name='tb_ppo', callback=checkpoint_callback)
model.save("ppo_usv")