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
        for init_mode in ["", "Random", "Buffered"]:
            gym.envs.register(
                id=f'Reach{init_mode}{control_mode}{vision_mode}-v0',
                entry_point='angorapy.tasks:Reach',
                kwargs={"relative_control": control_mode == "Relative", "vision": vision_mode == "Visual",
                        **({"initial_qpos": init_mode.lower()} if init_mode else {}),
                        "n_substeps": N_SUBSTEPS},
                max_episode_steps=SHADOWHAND_MAX_STEPS,
            )

            gym.envs.register(
                id=f'Reach{init_mode}{control_mode}{vision_mode}NoTouch-v0',
                entry_point='angorapy.tasks:Reach',
                kwargs={"relative_control": control_mode == "Relative", "vision": vision_mode == "Visual",
                        **({"initial_qpos": init_mode.lower()} if init_mode else {}), "touch": False,
                        "n_substeps": N_SUBSTEPS},
                max_episode_steps=SHADOWHAND_MAX_STEPS,
            )

            gym.envs.register(
                id=f'FreeReach{init_mode}{control_mode}{vision_mode}-v0',
                entry_point='angorapy.tasks:FreeReach',
                kwargs={"relative_control": control_mode == "Relative", "vision": vision_mode == "Visual",
                        **({"initial_qpos": init_mode.lower()} if init_mode else {}),
                        "n_substeps": N_SUBSTEPS},
                max_episode_steps=SHADOWHAND_MAX_STEPS,
            )

            for i, name in enumerate(["FF", "MF", "RF", "LF"]):
                gym.envs.register(
                    id=f'FreeReach{name}{init_mode}{control_mode}{vision_mode}-v0',
                    entry_point='angorapy.tasks:FreeReach',
                    kwargs={"relative_control": control_mode == "Relative",
                            "vision": vision_mode == "Visual",
                            "force_finger": i,
                            **({"initial_qpos": init_mode.lower()} if init_mode else {}),
                            "n_substeps": N_SUBSTEPS},
                    max_episode_steps=SHADOWHAND_MAX_STEPS,
                )

                gym.envs.register(
                    id=f'Reach{name}{init_mode}{control_mode}{vision_mode}-v0',
                    entry_point='angorapy.tasks:Reach',
                    kwargs={"relative_control": control_mode == "Relative",
                            "vision": vision_mode == "Visual",
                            "force_finger": i,
                            **({"initial_qpos": init_mode.lower()} if init_mode else {}),
                            "n_substeps": N_SUBSTEPS},
                    max_episode_steps=SHADOWHAND_MAX_STEPS,
                )

            # REACH SEQUENCES
            gym.envs.register(
                id=f'FreeReachSequential{init_mode}{control_mode}{vision_mode}-v0',
                entry_point='angorapy.tasks:FreeReachSequential',
                kwargs={"relative_control": control_mode == "Relative", "vision": vision_mode == "Visual",
                        **({"initial_qpos": init_mode.lower()} if init_mode else {}),
                        "n_substeps": N_SUBSTEPS},
                max_episode_steps=SHADOWHAND_SEQUENCE_MAX_STEPS,
            )

            gym.envs.register(
                id=f'ReachSequential{init_mode}{control_mode}{vision_mode}-v0',
                entry_point='angorapy.tasks:ReachSequential',
                kwargs={"relative_control": control_mode == "Relative", "vision": vision_mode == "Visual",
                        **({"initial_qpos": init_mode.lower()} if init_mode else {}),
                        "n_substeps": N_SUBSTEPS},
                max_episode_steps=SHADOWHAND_SEQUENCE_MAX_STEPS,
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
