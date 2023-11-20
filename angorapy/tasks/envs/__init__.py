"""Register all builtin tasks."""
import gymnasium as gym

from angorapy.common.const import N_SUBSTEPS
from angorapy.common.const import SHADOWHAND_MAX_STEPS
from angorapy.tasks.envs.adapted import HalfCheetahNoVelEnv
from angorapy.tasks.envs.adapted import InvertedPendulumNoVelEnv
from angorapy.tasks.envs.adapted import LunarLanderContinuousNoVel
from angorapy.tasks.envs.adapted import LunarLanderMultiDiscrete
from angorapy.tasks.envs.adapted import ReacherNoVelEnv
from angorapy.tasks.envs.cognitive.hanoi import HanoiEnv
from angorapy.tasks.envs.dexterity.manipulate import ManipulateBlock
from angorapy.tasks.envs.dexterity.manipulate import ManipulateBlockAsymmetric
from angorapy.tasks.envs.dexterity.manipulate import ManipulateBlockDiscrete
from angorapy.tasks.envs.dexterity.manipulate import ManipulateBlockDiscrete
from angorapy.tasks.envs.dexterity.manipulate import ManipulateBlockDiscreteAsymmetric
from angorapy.tasks.envs.dexterity.reach import FreeReach
from angorapy.tasks.envs.dexterity.reach import FreeReachSequential
from angorapy.tasks.envs.dexterity.reach import Reach
from angorapy.tasks.envs.dexterity.reach import ReachSequential

# REACHING
for vision_mode in ["Visual", ""]:
    for control_mode in ["Relative", "Absolute"]:
        gym.envs.register(
            id=f'Reach{control_mode}{vision_mode}-v0',
            entry_point='angorapy.tasks.envs.dexterity.reach:Reach',
            kwargs={"relative_control": control_mode == "Relative", "vision": vision_mode == "Visual",
                    "n_substeps": N_SUBSTEPS},
            max_episode_steps=SHADOWHAND_MAX_STEPS,
        )

        gym.envs.register(
            id=f'Reach{control_mode}{vision_mode}NoTouch-v0',
            entry_point='angorapy.tasks.envs.dexterity.reach:Reach',
            kwargs={"relative_control": control_mode == "Relative", "vision": vision_mode == "Visual", "touch": False,
                    "n_substeps": N_SUBSTEPS},
            max_episode_steps=SHADOWHAND_MAX_STEPS,
        )

        gym.envs.register(
            id=f'FreeReach{control_mode}{vision_mode}-v0',
            entry_point='angorapy.tasks.envs.dexterity.reach:FreeReach',
            kwargs={"relative_control": control_mode == "Relative", "vision": vision_mode == "Visual",
                    "n_substeps": N_SUBSTEPS},
            max_episode_steps=SHADOWHAND_MAX_STEPS,
        )

        for i, name in enumerate(["FF", "MF", "RF", "LF"]):
            gym.envs.register(
                id=f'FreeReach{name}{control_mode}{vision_mode}-v0',
                entry_point='angorapy.tasks.envs.dexterity.reach:FreeReach',
                kwargs={"relative_control": control_mode == "Relative",
                        "vision": vision_mode == "Visual",
                        "force_finger": i,
                        "n_substeps": N_SUBSTEPS},
                max_episode_steps=SHADOWHAND_MAX_STEPS,
            )

            gym.envs.register(
                id=f'Reach{name}{control_mode}{vision_mode}-v0',
                entry_point='angorapy.tasks.envs.dexterity.reach:Reach',
                kwargs={"relative_control": control_mode == "Relative",
                        "vision": vision_mode == "Visual",
                        "force_finger": i,
                        "n_substeps": N_SUBSTEPS},
                max_episode_steps=SHADOWHAND_MAX_STEPS,
            )

# MANIPULATE
gym.envs.register(
    id=f'ManipulateBlock-v0',
    entry_point='angorapy.tasks.envs.dexterity.manipulate:ManipulateBlock',
    kwargs={"delta_t": 0.008},
    max_episode_steps=50 * 100,
)

gym.envs.register(
    id=f'ManipulateBlockVisual-v0',
    entry_point='angorapy.tasks.envs.dexterity.manipulate:ManipulateBlock',
    kwargs={"delta_t": 0.008, "vision": True},
    max_episode_steps=50 * 100,
)

gym.envs.register(
    id=f'ManipulateBlockDiscrete-v0',
    entry_point='angorapy.tasks.envs.dexterity.manipulate:ManipulateBlockDiscrete',
    kwargs={"delta_t": 0.008},
    max_episode_steps=50 * 100,
)

gym.envs.register(
    id=f'ManipulateBlockVisualDiscrete-v0',
    entry_point='angorapy.tasks.envs.dexterity.manipulate:ManipulateBlockDiscrete',
    kwargs={"delta_t": 0.008, "vision": True},
    max_episode_steps=50 * 100,
)

gym.envs.register(
    id=f'ManipulateBlockDiscreteAsymmetric-v0',
    entry_point='angorapy.tasks.envs.dexterity.manipulate:ManipulateBlockDiscreteAsymmetric',
    kwargs={"delta_t": 0.008},
    max_episode_steps=50 * 100,
)

gym.envs.register(
    id=f'ManipulateBlockVisualDiscreteAsymmetric-v0',
    entry_point='angorapy.tasks.envs.dexterity.manipulate:ManipulateBlockDiscreteAsymmetric',
    kwargs={"delta_t": 0.008, "vision": True},
    max_episode_steps=50 * 100,
)

gym.envs.register(
    id=f'ManipulateBlockAsymmetric-v0',
    entry_point='angorapy.tasks.envs.dexterity.manipulate:ManipulateBlockAsymmetric',
    kwargs={"delta_t": 0.008},
    max_episode_steps=50 * 100,
)

gym.envs.register(
    id=f'ManipulateBlockVisualAsymmetric-v0',
    entry_point='angorapy.tasks.envs.dexterity.manipulate:ManipulateBlockAsymmetric',
    kwargs={"delta_t": 0.008, "vision": True},
    max_episode_steps=50 * 100,
)

# for backwards compatibility; TODO remove at 1.0
gym.envs.register(
    id=f'ManipulateBlockDiscreteAsynchronous-v0',
    entry_point='angorapy.tasks.envs.dexterity.manipulate:ManipulateBlockDiscreteAsymmetric',
    kwargs={"delta_t": 0.008},
    max_episode_steps=50 * 100,
)

gym.envs.register(
    id=f'ManipulateBlockVisualDiscreteAsynchronous-v0',
    entry_point='angorapy.tasks.envs.dexterity.manipulate:ManipulateBlockDiscreteAsymmetric',
    kwargs={"delta_t": 0.008, "vision": True},
    max_episode_steps=50 * 100,
)

gym.envs.register(
    id=f'ManipulateBlockAsynchronous-v0',
    entry_point='angorapy.tasks.envs.dexterity.manipulate:ManipulateBlockAsymmetric',
    kwargs={"delta_t": 0.008},
    max_episode_steps=50 * 100,
)

gym.envs.register(
    id=f'ManipulateBlockVisualAsynchronous-v0',
    entry_point='angorapy.tasks.envs.dexterity.manipulate:ManipulateBlockAsymmetric',
    kwargs={"delta_t": 0.008, "vision": True},
    max_episode_steps=50 * 100,
)

gym.envs.register(
    id="HanoiTower-v0",
    entry_point="angorapy.tasks.envs.cognitive.hanoi:HanoiEnv",
)
