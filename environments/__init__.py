"""Module for additional environments as well as registering modified environments."""

# MODIFIED ENVIRONMENTS
import gym

gym.envs.register(
    id='MountainCarLong-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=500,
    reward_threshold=-110.0,
)