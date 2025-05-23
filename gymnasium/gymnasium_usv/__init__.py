from gymnasium.envs.registration import make, register, registry, spec


register(
    id="js-collision-v0",
    entry_point="gymnasium_usv.envs:JsCollisionV0",
    max_episode_steps=4096,
)

register(
    id="usv-local-collision-avoidance-v0",
    entry_point="gymnasium_usv.envs:UsvLocalCollisionAvoidanceV0",
    max_episode_steps=4096,
)