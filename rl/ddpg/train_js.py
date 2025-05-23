import add_path
import torch as th
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import DDPG

from utils.feature_extractor.js_feature_extractor import JsFeatureExtractor
from utils.callback.ddpg_save_same_model_cb import DDPGSaveSameModelCB

from datetime import date


# Parallel environments
vec_env = make_vec_env("gymnasium_vrx:js-collision-v0", n_envs=1)

policy_kwargs = dict(
    features_extractor_class=JsFeatureExtractor,
    net_arch=(dict(pi=[512, 256, 256], qf=[512, 256, 256])),
    activation_fn = th.nn.ELU,
)

today = date.today()
checkpoint_callback = DDPGSaveSameModelCB(
    save_freq=50000,
    save_path="./logs/",
    name_prefix="ddpg_js_nav",
    save_replay_buffer=True,
    save_vecnormalize=True,
    overwrite=True,  # ✅ This enables overwrite mode
)
model = DDPG("MultiInputPolicy", vec_env, 
            verbose=1, 
            policy_kwargs=policy_kwargs, 
            batch_size=128,
            learning_rate=1e-6,
            tensorboard_log='tb_ddpg'
            )
            
model.learn(total_timesteps=20_000_000, tb_log_name='tb_ddpg', callback=checkpoint_callback)
model.save("ddpg_js_nav")