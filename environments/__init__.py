"""Module for additional environments as well as registering modified environments."""

import gym

from environments.adapted import InvertedPendulumNoVelEnv, ReacherNoVelEnv, HalfCheetahNoVelEnv, \
    LunarLanderContinuousNoVel
from environments.manipulate import ManipulateBlock, ManipulateBlockVector
from environments.reach import Reach, MultiReach, FreeReach, FreeReachSequential, ReachSequential
from common.const import SHADOWHAND_MAX_STEPS, SHADOWHAND_SEQUENCE_MAX_STEPS


# SHADOW HAND
gym.envs.register(
    id='BaseShadowHandEnv-v0',
    entry_point='environments:ManipulateBlock',
    kwargs={"visual_input": True, "max_steps": SHADOWHAND_MAX_STEPS},
)

gym.envs.register(
    id='ShadowHandBlind-v0',
    entry_point='environments:ManipulateBlock',
    kwargs={"visual_input": False, "max_steps": SHADOWHAND_MAX_STEPS},
)

# REACHING

for vision_mode in ["Visual", ""]:
    for control_mode in ["Relative", "Absolute"]:
        for init_mode in ["", "Random", "Buffered"]:
            gym.envs.register(
                id=f'Reach{init_mode}{control_mode}{vision_mode}-v0',
                entry_point='environments:Reach',
                kwargs={"relative_control": control_mode == "Relative", "vision": vision_mode == "Visual",
                        **({"initial_qpos": init_mode.lower()} if init_mode else {})},
                max_episode_steps=SHADOWHAND_MAX_STEPS,
            )

            gym.envs.register(
                id=f'FreeReach{init_mode}{control_mode}{vision_mode}-v0',
                entry_point='environments:FreeReach',
                kwargs={"relative_control": control_mode == "Relative", "vision": vision_mode == "Visual",
                        **({"initial_qpos": init_mode.lower()} if init_mode else {})},
                max_episode_steps=SHADOWHAND_MAX_STEPS,
            )

            for i, name in enumerate(["FF", "MF", "RF", "LF"]):
                gym.envs.register(
                    id=f'FreeReach{name}{init_mode}{control_mode}{vision_mode}-v0',
                    entry_point='environments:FreeReach',
                    kwargs={"relative_control": control_mode == "Relative", "vision": vision_mode == "Visual", "force_finger": i,
                            **({"initial_qpos": init_mode.lower()} if init_mode else {})},
                    max_episode_steps=SHADOWHAND_MAX_STEPS,
                )

            # REACH SEQUENCES

            gym.envs.register(
                id=f'FreeReachSequential{init_mode}{control_mode}{vision_mode}-v0',
                entry_point='environments:FreeReachSequential',
                kwargs={"relative_control": control_mode == "Relative", "vision": vision_mode == "Visual",
                        **({"initial_qpos": init_mode.lower()} if init_mode else {})},
                max_episode_steps=SHADOWHAND_SEQUENCE_MAX_STEPS,
            )

            gym.envs.register(
                id=f'ReachSequential{init_mode}{control_mode}{vision_mode}-v0',
                entry_point='environments:ReachSequential',
                kwargs={"relative_control": control_mode == "Relative", "vision": vision_mode == "Visual",
                        **({"initial_qpos": init_mode.lower()} if init_mode else {})},
                max_episode_steps=SHADOWHAND_SEQUENCE_MAX_STEPS,
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
