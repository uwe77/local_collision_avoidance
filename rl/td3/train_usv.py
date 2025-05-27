import add_path
import torch as th
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import TD3

from utils.callback.td3_save_same_model_cb import TD3SaveSameModelCB
from utils.feature_extractor.usv_feature_extractor import USVFeatureExtractor
from datetime import date


# Parallel environments
vec_env = make_vec_env(
    "gymnasium_usv:usv-local-collision-avoidance-v0", 
    env_kwargs={
        "render_mode": "none",
        "usv_name": "js",
        "enable_obstacle": False,
        "obstacle_max_speed": 5.0,
        "reset_range": 100.0,},
    n_envs=1
    )

policy_kwargs = dict(
    features_extractor_class=USVFeatureExtractor,
    activation_fn = th.nn.ELU,
)

today = date.today()
checkpoint_callback = TD3SaveSameModelCB(
    save_freq=50000,
    save_path="./logs/",
    name_prefix="td3_usv",
    save_replay_buffer=True,
    save_vecnormalize=True,
    overwrite=True,  # ✅ This enables overwrite mode
)

model = TD3("MultiInputPolicy", vec_env, 
            verbose=1, 
            policy_kwargs=policy_kwargs, 
            batch_size=64,
            learning_rate=1e-7,
            tensorboard_log='tb_td3'
            )
            
model.learn(total_timesteps=20_000_000, tb_log_name='tb_td3', callback=checkpoint_callback)
model.save("td3_usv")