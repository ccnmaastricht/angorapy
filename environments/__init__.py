"""Module for additional environments as well as registering modified environments."""

import gym

from environments.adapted import InvertedPendulumNoVelEnv, ReacherNoVelEnv, HalfCheetahNoVelEnv, \
    LunarLanderContinuousNoVel
from environments.shadowhand import ShadowHandBlock, ShadowHandReach, ShadowHandBlockVector, ShadowHandMultiReach, \
    ShadowHandFreeReach, ShadowHandTappingSequence, ShadowHandDelayedTappingSequence

# SHADOW HAND

gym.envs.register(
    id='ShadowHand-v0',
    entry_point='environments:ShadowHandBlock',
    kwargs={"visual_input": True, "max_steps": 100},
)

gym.envs.register(
    id='ShadowHandBlind-v0',
    entry_point='environments:ShadowHandBlock',
    kwargs={"visual_input": False, "max_steps": 500},
)

# REACH

gym.envs.register(
    id='HandReachDenseRelative-v0',
    entry_point='environments:ShadowHandReach',
    kwargs={"reward_type": "dense", "relative_control": True},
    max_episode_steps=100,
)

gym.envs.register(
    id='HandReachDenseRelative-v1',
    entry_point='environments:ShadowHandReach',
    kwargs={"reward_type": "dense", "relative_control": True, "success_multiplier": 0.1},
    max_episode_steps=100,
)

gym.envs.register(
    id='HandReachDenseAbsolute-v0',
    entry_point='environments:ShadowHandReach',
    kwargs={"reward_type": "dense", "relative_control": False},
    max_episode_steps=100,
)

gym.envs.register(
    id='HandReachDenseAbsolute-v1',
    entry_point='environments:ShadowHandReach',
    kwargs={"reward_type": "dense", "relative_control": False, "success_multiplier": 0.1},
    max_episode_steps=100,
)

gym.envs.register(
    id='MultiReachAbsolute-v0',
    entry_point='environments:ShadowHandMultiReach',
    kwargs={"reward_type": "dense", "relative_control": False, "success_multiplier": 0.1},
    max_episode_steps=100,
)

# FREE REACHING

gym.envs.register(
    id='HandFreeReachRelative-v0',
    entry_point='environments:ShadowHandFreeReach',
    kwargs={"relative_control": True, "success_multiplier": 0.1},
    max_episode_steps=100,
)

gym.envs.register(
    id='HandFreeReachAbsolute-v0',
    entry_point='environments:ShadowHandFreeReach',
    kwargs={"relative_control": False, "success_multiplier": 0.1},
    max_episode_steps=100,
)

for i, name in enumerate(["FF", "MF", "RF", "LF"]):
    gym.envs.register(
        id=f'HandFreeReach{name}Absolute-v0',
        entry_point='environments:ShadowHandFreeReach',
        kwargs={"relative_control": False, "success_multiplier": 0.1, "force_finger": i},
        max_episode_steps=100,
    )

# HAND TAPPING

gym.envs.register(
    id='HandTappingAbsolute-v0',
    entry_point='environments:ShadowHandTappingSequence',
    kwargs={"relative_control": False, "success_multiplier": 1},
    max_episode_steps=200,
)

gym.envs.register(
    id='HandTappingAbsolute-v1',
    entry_point='environments:ShadowHandDelayedTappingSequence',
    kwargs={"relative_control": False},
    max_episode_steps=200,
)

# MANIPULATE

gym.envs.register(
    id='EasyBlockManipulate-v0',
    entry_point='environments:ShadowHandBlockVector',
    kwargs={'target_position': 'ignore', 'target_rotation': 'xyz', "reward_type": "dense"},
    max_episode_steps=100,
)

# MODIFIED ENVIRONMENTS

gym.envs.register(
    id='MountainCarLong-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=500,
    reward_threshold=-110.0,
)

gym.envs.register(
    id="InvertedPendulumNoVel-v2",
    entry_point="environments:InvertedPendulumNoVelEnv",
    max_episode_steps=1000,
    reward_threshold=950.0,
)

gym.envs.register(
    id='ReacherNoVel-v2',
    entry_point='environments:ReacherNoVelEnv',
    max_episode_steps=50,
    reward_threshold=-3.75,
)

gym.envs.register(
    id='HalfCheetahNoVel-v2',
    entry_point='environments:HalfCheetahNoVelEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

gym.envs.register(
    id='LunarLanderContinuousNoVel-v2',
    entry_point='environments:LunarLanderContinuousNoVel',
    max_episode_steps=1000,
    reward_threshold=200,
)
