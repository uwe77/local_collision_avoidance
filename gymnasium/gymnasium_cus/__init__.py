from gymnasium.envs.registration import make, register, registry, spec


register(
    id="js-collision-v0",
    entry_point="gymnasium_cus.envs:JsCollisionV0",
    max_episode_steps=4096,
)

register(
    id="js-teacher-student-v0",
    entry_point="gymnasium_cus.envs:JSTeacherStudentV0",
    max_episode_steps=4096,
)

register(
    id="js-multiple-missions-v0",
    entry_point="gymnasium_cus.envs:JSMultipleMissionsV0",
    max_episode_steps=4096,
)