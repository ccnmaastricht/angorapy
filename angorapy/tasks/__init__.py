"""Module for additional environments as well as registering modified environments."""

import gymnasium as gym

from angorapy.common.const import SHADOWHAND_MAX_STEPS, SHADOWHAND_SEQUENCE_MAX_STEPS, N_SUBSTEPS
from angorapy.tasks.core import AnthropomorphicEnv
from angorapy.tasks.envs.adapted import InvertedPendulumNoVelEnv, ReacherNoVelEnv, HalfCheetahNoVelEnv, \
    LunarLanderContinuousNoVel, LunarLanderMultiDiscrete
from angorapy.tasks.envs.dexterity.manipulate import ManipulateBlock, ManipulateBlockDiscrete, HumanoidManipulateBlock, \
    HumanoidManipulateBlockDiscrete, HumanoidManipulateBlockDiscreteAsynchronous, HumanoidManipulateBlockAsynchronous
from angorapy.tasks.envs.dexterity.reach import Reach, FreeReach, FreeReachSequential, ReachSequential

# SHADOW HAND
gym.envs.register(
    id='BaseShadowHandEnv-v0',
    entry_point='angorapy.tasks:ManipulateBlock',
    kwargs={"visual_input": True, "max_steps": SHADOWHAND_MAX_STEPS},
)

gym.envs.register(
    id='ShadowHandBlind-v0',
    entry_point='angorapy.tasks:ManipulateBlock',
    kwargs={"visual_input": False, "max_steps": SHADOWHAND_MAX_STEPS},
)

# REACHING

gym.envs.register(
    id='HandReachDenseAbsolute-v1',
    entry_point='angorapy.tasks:OldShadowHandReach',
    kwargs={"reward_type": "dense", "relative_control": False, "success_multiplier": 0.1},
    max_episode_steps=100,
)

gym.envs.register(
    id='HandReachDenseAbsoluteFine-v1',
    entry_point='angorapy.tasks:OldShadowHandReach',
    kwargs={"reward_type": "dense", "n_substeps": 1, "relative_control": False, "success_multiplier": 0.1},
    max_episode_steps=100,
)

for step_granularity in ["Fine", ""]:
    for vision_mode in ["Visual", ""]:
        for control_mode in ["Relative", "Absolute"]:
            for init_mode in ["", "Random", "Buffered"]:
                gym.envs.register(
                    id=f'Reach{init_mode}{control_mode}{vision_mode}{step_granularity}-v0',
                    entry_point='angorapy.tasks:Reach',
                    kwargs={"relative_control": control_mode == "Relative", "vision": vision_mode == "Visual",
                            **({"initial_qpos": init_mode.lower()} if init_mode else {}),
                            "n_substeps": 1 if step_granularity == "Fine" else N_SUBSTEPS},
                    max_episode_steps=SHADOWHAND_MAX_STEPS,
                )

                gym.envs.register(
                    id=f'Reach{init_mode}{control_mode}{vision_mode}NoTouch{step_granularity}-v0',
                    entry_point='angorapy.tasks:Reach',
                    kwargs={"relative_control": control_mode == "Relative", "vision": vision_mode == "Visual",
                            **({"initial_qpos": init_mode.lower()} if init_mode else {}), "touch": False,
                            "n_substeps": 1 if step_granularity == "Fine" else N_SUBSTEPS},
                    max_episode_steps=SHADOWHAND_MAX_STEPS,
                )

                gym.envs.register(
                    id=f'FreeReach{init_mode}{control_mode}{vision_mode}{step_granularity}-v0',
                    entry_point='angorapy.tasks:FreeReach',
                    kwargs={"relative_control": control_mode == "Relative", "vision": vision_mode == "Visual",
                            **({"initial_qpos": init_mode.lower()} if init_mode else {}),
                            "n_substeps": 1 if step_granularity == "Fine" else N_SUBSTEPS},
                    max_episode_steps=SHADOWHAND_MAX_STEPS,
                )

                for i, name in enumerate(["FF", "MF", "RF", "LF"]):
                    gym.envs.register(
                        id=f'FreeReach{name}{init_mode}{control_mode}{vision_mode}{step_granularity}-v0',
                        entry_point='angorapy.tasks:FreeReach',
                        kwargs={"relative_control": control_mode == "Relative",
                                "vision": vision_mode == "Visual",
                                "force_finger": i,
                                **({"initial_qpos": init_mode.lower()} if init_mode else {}),
                                "n_substeps": 1 if step_granularity == "Fine" else N_SUBSTEPS},
                        max_episode_steps=SHADOWHAND_MAX_STEPS,
                    )

                    gym.envs.register(
                        id=f'Reach{name}{init_mode}{control_mode}{vision_mode}{step_granularity}-v0',
                        entry_point='angorapy.tasks:Reach',
                        kwargs={"relative_control": control_mode == "Relative",
                                "vision": vision_mode == "Visual",
                                "force_finger": i,
                                **({"initial_qpos": init_mode.lower()} if init_mode else {}),
                                "n_substeps": 1 if step_granularity == "Fine" else N_SUBSTEPS},
                        max_episode_steps=SHADOWHAND_MAX_STEPS,
                    )

                # REACH SEQUENCES
                gym.envs.register(
                    id=f'FreeReachSequential{init_mode}{control_mode}{vision_mode}{step_granularity}-v0',
                    entry_point='angorapy.tasks:FreeReachSequential',
                    kwargs={"relative_control": control_mode == "Relative", "vision": vision_mode == "Visual",
                            **({"initial_qpos": init_mode.lower()} if init_mode else {}),
                            "n_substeps": 1 if step_granularity == "Fine" else N_SUBSTEPS},
                    max_episode_steps=SHADOWHAND_SEQUENCE_MAX_STEPS,
                )

                gym.envs.register(
                    id=f'ReachSequential{init_mode}{control_mode}{vision_mode}{step_granularity}-v0',
                    entry_point='angorapy.tasks:ReachSequential',
                    kwargs={"relative_control": control_mode == "Relative", "vision": vision_mode == "Visual",
                            **({"initial_qpos": init_mode.lower()} if init_mode else {}),
                            "n_substeps": 1 if step_granularity == "Fine" else N_SUBSTEPS},
                    max_episode_steps=SHADOWHAND_SEQUENCE_MAX_STEPS,
                )

# MANIPULATE
for control_mode in ["Relative", "Absolute"]:
    gym.envs.register(
        id=f'ManipulateBlock{control_mode}-v0',
        entry_point='angorapy.tasks:ManipulateBlock',
        kwargs={'target_position': 'ignore', 'target_rotation': 'xyz', "relative_control": control_mode == "Relative"},
        max_episode_steps=SHADOWHAND_MAX_STEPS,
    )

    gym.envs.register(
        id=f'ManipulateBlockDiscrete{control_mode}-v0',
        entry_point='angorapy.tasks:ManipulateBlockDiscrete',
        kwargs={'target_position': 'ignore',
                'target_rotation': 'xyz',
                "relative_control": control_mode == "Relative"},
        max_episode_steps=SHADOWHAND_MAX_STEPS,
    )

    gym.envs.register(
        id=f'ManipulateEgg{control_mode}-v0',
        entry_point='angorapy.tasks:ManipulateEgg',
        kwargs={'target_position': 'ignore', 'target_rotation': 'xyz', "relative_control": control_mode == "Relative"},
        max_episode_steps=SHADOWHAND_MAX_STEPS,
    )

gym.envs.register(
    id=f'OpenAIManipulate-v0',
    entry_point='angorapy.tasks:OpenAIManipulate',
    kwargs={},
    max_episode_steps=50 * 100,
)

gym.envs.register(
    id=f'OpenAIManipulateDiscrete-v0',
    entry_point='angorapy.tasks:OpenAIManipulateDiscrete',
    kwargs={},
    max_episode_steps=50 * 100,
)

gym.envs.register(
    id=f'OpenAIManipulateApproxDiscrete-v0',
    entry_point='angorapy.tasks:OpenAIManipulateDiscrete',
    kwargs={"delta_t": 0.008},
    max_episode_steps=50 * 100,
)

gym.envs.register(
    id=f'OpenAIManipulateApprox-v0',
    entry_point='angorapy.tasks:OpenAIManipulate',
    kwargs={"delta_t": 0.008},
    max_episode_steps=50 * 100,
)

gym.envs.register(
    id=f'ManipulateBlockApproxDiscrete-v0',
    entry_point='angorapy.tasks:ManipulateBlockDiscrete',
    kwargs={"delta_t": 0.008},
    max_episode_steps=50 * 100,
)

gym.envs.register(
    id=f'HumanoidManipulateBlockDiscrete-v0',
    entry_point='angorapy.tasks:HumanoidManipulateBlockDiscrete',
    kwargs={"delta_t": 0.008},
    max_episode_steps=50 * 100,
)

gym.envs.register(
    id=f'HumanoidManipulateBlockDiscreteAsynchronous-v0',
    entry_point='angorapy.tasks:HumanoidManipulateBlockDiscreteAsynchronous',
    kwargs={"delta_t": 0.008},
    max_episode_steps=50 * 100,
)

gym.envs.register(
    id=f'HumanoidVisualManipulateBlockDiscreteAsynchronous-v0',
    entry_point='angorapy.tasks:HumanoidManipulateBlockDiscreteAsynchronous',
    kwargs={"delta_t": 0.008, "vision": True},
    max_episode_steps=50 * 100,
)

gym.envs.register(
    id=f'HumanoidManipulateBlockAsynchronous-v0',
    entry_point='angorapy.tasks:HumanoidManipulateBlockAsynchronous',
    kwargs={"delta_t": 0.008},
    max_episode_steps=50 * 100,
)

gym.envs.register(
    id=f'HumanoidManipulateBlock-v0',
    entry_point='angorapy.tasks:HumanoidManipulateBlock',
    kwargs={"delta_t": 0.008},
    max_episode_steps=50 * 100,
)

gym.envs.register(
    id=f'HumanoidVisualManipulateBlock-v0',
    entry_point='angorapy.tasks:HumanoidManipulateBlock',
    kwargs={"delta_t": 0.008, "vision": True},
    max_episode_steps=50 * 100,
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
    entry_point="angorapy.tasks:InvertedPendulumNoVelEnv",
    max_episode_steps=1000,
    reward_threshold=950.0,
)

gym.envs.register(
    id='ReacherNoVel-v2',
    entry_point='angorapy.tasks:ReacherNoVelEnv',
    max_episode_steps=50,
    reward_threshold=-3.75,
)

gym.envs.register(
    id='HalfCheetahNoVel-v2',
    entry_point='angorapy.tasks:HalfCheetahNoVelEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

gym.envs.register(
    id='LunarLanderContinuousNoVel-v2',
    entry_point='angorapy.tasks:LunarLanderContinuousNoVel',
    max_episode_steps=1000,
    reward_threshold=200,
)

gym.envs.register(
    id='LunarLanderMultiDiscrete-v2',
    entry_point='angorapy.tasks:LunarLanderMultiDiscrete',
    max_episode_steps=1000,
    reward_threshold=200,
)

gym.envs.register(
    id="HanoiTower-v0",
    entry_point="angorapy.tasks:HanoiEnv",
)

__all__ = [
    "AnthropomorphicEnv",
]
