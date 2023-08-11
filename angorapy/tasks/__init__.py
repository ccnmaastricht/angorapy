"""Module for additional environments as well as registering modified environments."""

import gymnasium as gym

from angorapy.common.const import SHADOWHAND_MAX_STEPS, SHADOWHAND_SEQUENCE_MAX_STEPS, N_SUBSTEPS
from angorapy.tasks.core import AnthropomorphicEnv
from angorapy.tasks.envs.adapted import InvertedPendulumNoVelEnv, ReacherNoVelEnv, HalfCheetahNoVelEnv, \
    LunarLanderContinuousNoVel, LunarLanderMultiDiscrete
from angorapy.tasks.envs.dexterity.manipulate import ManipulateBlock, ManipulateBlockDiscrete, \
    ManipulateBlockDiscrete, ManipulateBlockDiscreteAsymmetric, ManipulateBlockAsymmetric
from angorapy.tasks.envs.dexterity.reach import Reach, FreeReach, FreeReachSequential, ReachSequential
from angorapy.tasks.envs.cognitive.hanoi import HanoiEnv


# REACHING
for vision_mode in ["Visual", ""]:
    for control_mode in ["Relative", "Absolute"]:
        gym.envs.register(
            id=f'Reach{control_mode}{vision_mode}-v0',
            entry_point='angorapy.tasks:Reach',
            kwargs={"relative_control": control_mode == "Relative", "vision": vision_mode == "Visual",
                    "n_substeps": N_SUBSTEPS},
            max_episode_steps=SHADOWHAND_MAX_STEPS,
        )

        gym.envs.register(
            id=f'Reach{control_mode}{vision_mode}NoTouch-v0',
            entry_point='angorapy.tasks:Reach',
            kwargs={"relative_control": control_mode == "Relative", "vision": vision_mode == "Visual", "touch": False,
                    "n_substeps": N_SUBSTEPS},
            max_episode_steps=SHADOWHAND_MAX_STEPS,
        )

        gym.envs.register(
            id=f'FreeReach{control_mode}{vision_mode}-v0',
            entry_point='angorapy.tasks:FreeReach',
            kwargs={"relative_control": control_mode == "Relative", "vision": vision_mode == "Visual",
                    "n_substeps": N_SUBSTEPS},
            max_episode_steps=SHADOWHAND_MAX_STEPS,
        )

        for i, name in enumerate(["FF", "MF", "RF", "LF"]):
            gym.envs.register(
                id=f'FreeReach{name}{control_mode}{vision_mode}-v0',
                entry_point='angorapy.tasks:FreeReach',
                kwargs={"relative_control": control_mode == "Relative",
                        "vision": vision_mode == "Visual",
                        "force_finger": i,
                        "n_substeps": N_SUBSTEPS},
                max_episode_steps=SHADOWHAND_MAX_STEPS,
            )

            gym.envs.register(
                id=f'Reach{name}{control_mode}{vision_mode}-v0',
                entry_point='angorapy.tasks:Reach',
                kwargs={"relative_control": control_mode == "Relative",
                        "vision": vision_mode == "Visual",
                        "force_finger": i,
                        "n_substeps": N_SUBSTEPS},
                max_episode_steps=SHADOWHAND_MAX_STEPS,
            )


# MANIPULATE
gym.envs.register(
    id=f'ManipulateBlock-v0',
    entry_point='angorapy.tasks:ManipulateBlock',
    kwargs={"delta_t": 0.008},
    max_episode_steps=50 * 100,
)

gym.envs.register(
    id=f'ManipulateBlockVisual-v0',
    entry_point='angorapy.tasks:ManipulateBlock',
    kwargs={"delta_t": 0.008, "vision": True},
    max_episode_steps=50 * 100,
)

gym.envs.register(
    id=f'ManipulateBlockDiscrete-v0',
    entry_point='angorapy.tasks:ManipulateBlockDiscrete',
    kwargs={"delta_t": 0.008},
    max_episode_steps=50 * 100,
)

gym.envs.register(
    id=f'ManipulateBlockVisualDiscrete-v0',
    entry_point='angorapy.tasks:ManipulateBlockDiscrete',
    kwargs={"delta_t": 0.008, "vision": True},
    max_episode_steps=50 * 100,
)

gym.envs.register(
    id=f'ManipulateBlockDiscreteAsymmetric-v0',
    entry_point='angorapy.tasks:ManipulateBlockDiscreteAsymmetric',
    kwargs={"delta_t": 0.008},
    max_episode_steps=50 * 100,
)

gym.envs.register(
    id=f'ManipulateBlockVisualDiscreteAsymmetric-v0',
    entry_point='angorapy.tasks:ManipulateBlockDiscreteAsymmetric',
    kwargs={"delta_t": 0.008, "vision": True},
    max_episode_steps=50 * 100,
)

gym.envs.register(
    id=f'ManipulateBlockAsymmetric-v0',
    entry_point='angorapy.tasks:ManipulateBlockAsymmetric',
    kwargs={"delta_t": 0.008},
    max_episode_steps=50 * 100,
)

gym.envs.register(
    id=f'ManipulateBlockVisualAsymmetric-v0',
    entry_point='angorapy.tasks:ManipulateBlockAsymmetric',
    kwargs={"delta_t": 0.008, "vision": True},
    max_episode_steps=50 * 100,
)


# for backwards compatibility; TODO remove at 1.0
gym.envs.register(
    id=f'ManipulateBlockDiscreteAsynchronous-v0',
    entry_point='angorapy.tasks:ManipulateBlockDiscreteAsymmetric',
    kwargs={"delta_t": 0.008},
    max_episode_steps=50 * 100,
)

gym.envs.register(
    id=f'ManipulateBlockVisualDiscreteAsynchronous-v0',
    entry_point='angorapy.tasks:ManipulateBlockDiscreteAsymmetric',
    kwargs={"delta_t": 0.008, "vision": True},
    max_episode_steps=50 * 100,
)

gym.envs.register(
    id=f'ManipulateBlockAsynchronous-v0',
    entry_point='angorapy.tasks:ManipulateBlockAsymmetric',
    kwargs={"delta_t": 0.008},
    max_episode_steps=50 * 100,
)

gym.envs.register(
    id=f'ManipulateBlockVisualAsynchronous-v0',
    entry_point='angorapy.tasks:ManipulateBlockAsymmetric',
    kwargs={"delta_t": 0.008, "vision": True},
    max_episode_steps=50 * 100,
)



gym.envs.register(
    id="HanoiTower-v0",
    entry_point="angorapy.tasks:HanoiEnv",
)

__all__ = [
    "AnthropomorphicEnv",
    "world_building"
]
