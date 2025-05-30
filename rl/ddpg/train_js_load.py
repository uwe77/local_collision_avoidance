import add_path
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import DDPG
from utils.callback.ddpg_save_same_model_cb import DDPGSaveSameModelCB

from datetime import date


# Parallel environments
vec_env = make_vec_env("gymnasium_vrx:js-collision-v0", n_envs=1)


today = date.today()
checkpoint_callback = DDPGSaveSameModelCB(
    save_freq=50000,
    save_path="./logs/",
    name_prefix="ddpg_js_nav",
    save_replay_buffer=True,
    save_vecnormalize=True,
    overwrite=True,
)

model = DDPG.load("logs/ddpg_js_nav_only")
model.learning_rate = 5e-6

model.set_env(vec_env)
model.learn(total_timesteps=20_000_000, tb_log_name='tb_ddpg', callback=checkpoint_callback)
model.save("ddpg_js_nav")