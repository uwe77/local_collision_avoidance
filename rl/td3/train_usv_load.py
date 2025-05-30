import add_path
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import TD3
from utils.callback.td3_save_same_model_cb import TD3SaveSameModelCB
from datetime import date


# Parallel environments
vec_env = make_vec_env(
    "gymnasium_usv:usv-local-collision-avoidance-v0", 
    env_kwargs={
        "render_mode": "none",
        "usv_name": "js",
        "enable_obstacle": False,
        "obstacle_max_speed": 5.0,
        "reset_range": 200.0,},
    n_envs=1
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
model = TD3.load("logs/td3_usv")
model.entropy_coef = 0.01
model.learning_rate = 1e-4

model.set_env(vec_env)
model.learn(total_timesteps=20_000_000, tb_log_name='tb_td3', callback=checkpoint_callback)
model.save("td3_usv")