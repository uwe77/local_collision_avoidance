import add_path
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from utils.callback.ppo_save_same_model_cb import PPOSaveSameModelCB
from datetime import date


# Parallel environments
max_steps = 1024
vec_env = make_vec_env(
    "gymnasium_usv:usv-local-collision-avoidance-v0", 
    env_kwargs={
        "render_mode": "none",
        "usv_name": "js",
        "enable_obstacle": False,
        "obstacle_max_speed": 5.0,
        "reset_range": 200.0,
        "max_steps": max_steps,},
    n_envs=1
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
model = PPO.load("logs/ppo_usv")
model.entropy_coef = 0.01
model.learning_rate = 1e-6

model.set_env(vec_env)
model.learn(total_timesteps=20_000_000, tb_log_name='tb_ppo', callback=checkpoint_callback)
model.save("ppo_usv")