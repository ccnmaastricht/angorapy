"""Module for additional environments as well as registering modified environments."""

import gym

from environments.adapted import InvertedPendulumNoVelEnv, ReacherNoVelEnv, HalfCheetahNoVelEnv, \
   LunarLanderContinuousNoVel
from environments.evasion import Evasion
from environments.evasionwalls import EvasionWalls
from environments.race import Race
from environments.shadowhand import ShadowHandBlock, ShadowHandReach, ShadowHandBlockVector, ShadowHandMultiReach
from environments.tunnel import Tunnel

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

# CUSTOM SIMPLE GAMES FROM RLASPA PROJECT

gym.envs.register(
    id='Race-v0',
    entry_point='environments:Race',
    kwargs={'width': 30, 'height': 30,
            'driver_chance': 0.05},
)

gym.envs.register(
    id='Evasion-v0',
    entry_point='environments:Evasion',
    kwargs={'width': 30, 'height': 30,
            'obstacle_chance': 0.05},
)

gym.envs.register(
    id='Tunnel-v0',
    entry_point='environments:Tunnel',
    kwargs={'width': 30, 'height': 30},
    reward_threshold=4950,
    max_episode_steps=500,
)

gym.envs.register(
    id='TunnelFlat-v0',
    entry_point='environments:Tunnel',
    kwargs={'width': 30, 'height': 30, 'mode': 'flat'},
    reward_threshold=4950,
    max_episode_steps=500,
)

gym.envs.register(
    id='TunnelTwoRows-v0',
    entry_point='environments:Tunnel',
    kwargs={'width': 30, 'height': 30, 'mode': 'rows'},
    reward_threshold=4950,
    max_episode_steps=500,
)

gym.envs.register(
    id='TunnelRAM-v0',
    entry_point='environments:Tunnel',
    kwargs={'width': 30, 'height': 30, "mode": "ram"},
    reward_threshold=4950,
    max_episode_steps=500,
)

gym.envs.register(
    id='EvasionWalls-v0',
    entry_point='environments:EvasionWalls',
    kwargs={'width': 30, 'height': 30},
)
