"""Module for additional environments as well as registering modified environments."""

import gym

from environments.adapted import InvertedPendulumNoVelEnv, ReacherNoVelEnv, HalfCheetahNoVelEnv, \
    LunarLanderContinuousNoVel
from environments.manipulate import ManipulateBlock, ManipulateBlockVector
from environments.reach import Reach, MultiReach, FreeReach, FreeReachVisual, \
    ShadowHandTappingSequence, ShadowHandDelayedTappingSequence, FreeReachSequential

# SHADOW HAND
from utilities.const import SHADOWHAND_MAX_STEPS

gym.envs.register(
    id='BaseShadowHand-v0',
    entry_point='environments:ManipulateBlock',
    kwargs={"visual_input": True, "max_steps": SHADOWHAND_MAX_STEPS},
)

gym.envs.register(
    id='ShadowHandBlind-v0',
    entry_point='environments:ManipulateBlock',
    kwargs={"visual_input": False, "max_steps": SHADOWHAND_MAX_STEPS},
)

# REACH

gym.envs.register(
    id='ReachDenseRelative-v0',
    entry_point='environments:Reach',
    kwargs={"reward_type": "dense", "relative_control": True},
    max_episode_steps=SHADOWHAND_MAX_STEPS,
)

gym.envs.register(
    id='ReachDenseRelative-v1',
    entry_point='environments:Reach',
    kwargs={"reward_type": "dense", "relative_control": True, "success_multiplier": 0.1},
    max_episode_steps=SHADOWHAND_MAX_STEPS,
)

gym.envs.register(
    id='ReachDenseAbsolute-v0',
    entry_point='environments:Reach',
    kwargs={"reward_type": "dense", "relative_control": False},
    max_episode_steps=SHADOWHAND_MAX_STEPS,
)

gym.envs.register(
    id='ReachDenseAbsolute-v1',
    entry_point='environments:Reach',
    kwargs={"reward_type": "dense", "relative_control": False, "success_multiplier": 0.1},
    max_episode_steps=SHADOWHAND_MAX_STEPS,
)

gym.envs.register(
    id='MultiReachAbsolute-v0',
    entry_point='environments:MultiReach',
    kwargs={"reward_type": "dense", "relative_control": False, "success_multiplier": 0.1},
    max_episode_steps=SHADOWHAND_MAX_STEPS,
)

# FREE REACHING

gym.envs.register(
    id='FreeReachRelative-v0',
    entry_point='environments:FreeReach',
    kwargs={"relative_control": True, "success_multiplier": 0.1},
    max_episode_steps=SHADOWHAND_MAX_STEPS,
)

gym.envs.register(
    id='FreeReachAbsolute-v0',
    entry_point='environments:FreeReach',
    kwargs={"relative_control": False, "success_multiplier": 0.1, "initial_qpos": "random"},
    max_episode_steps=SHADOWHAND_MAX_STEPS,
)

gym.envs.register(
    id='FreeReachVisualAbsolute-v0',
    entry_point='environments:FreeReachVisual',
    kwargs={"relative_control": False, "success_multiplier": 0.1},
    max_episode_steps=SHADOWHAND_MAX_STEPS,
)

gym.envs.register(
    id='FreeReachVisualRelative-v0',
    entry_point='environments:FreeReachVisual',
    kwargs={"relative_control": True, "success_multiplier": 0.1},
    max_episode_steps=SHADOWHAND_MAX_STEPS,
)

for i, name in enumerate(["FF", "MF", "RF", "LF"]):
    gym.envs.register(
        id=f'FreeReach{name}Absolute-v0',
        entry_point='environments:FreeReach',
        kwargs={"relative_control": False, "success_multiplier": 0.1, "force_finger": i},
        max_episode_steps=SHADOWHAND_MAX_STEPS,
    )

# HAND TAPPING

gym.envs.register(
    id='HandTappingAbsolute-v0',
    entry_point='environments:ShadowHandTappingSequence',
    kwargs={"relative_control": False, "success_multiplier": 1},
    max_episode_steps=SHADOWHAND_MAX_STEPS,
)

gym.envs.register(
    id='HandTappingAbsolute-v1',
    entry_point='environments:ShadowHandDelayedTappingSequence',
    kwargs={"relative_control": False},
    max_episode_steps=SHADOWHAND_MAX_STEPS,
)

gym.envs.register(
    id='FreeReachSequentialAbsolute-v0',
    entry_point='environments:FreeReachSequential',
    kwargs={"relative_control": False, "success_multiplier": 1, "initial_qpos": "random"},
    max_episode_steps=512,
)

# MANIPULATE

gym.envs.register(
    id='EasyBlockManipulate-v0',
    entry_point='environments:ShadowHandBlockVector',
    kwargs={'target_position': 'ignore', 'target_rotation': 'xyz', "reward_type": "dense"},
    max_episode_steps=SHADOWHAND_MAX_STEPS,
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
