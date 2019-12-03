"""Module for additional environments as well as registering modified environments."""
import gym

from environments.evasion import Evasion
from environments.evasionwalls import EvasionWalls
from environments.race import Race
from environments.tunnel import Tunnel, TunnelRAM
#from environments.shadowhand import ShadowHandBlock

# SHADOW HAND #

#gym.envs.register(
 #   id='ShadowHand-v0',
 #   entry_point='environments:ShadowHandBlock',
 #   kwargs={"visual_input": False},
#)

#gym.envs.register(
#   id='ShadowHand-v1',
#  entry_point='environments:ShadowHandBlock',
 #  kwargs={"visual_input": True, "max_steps": 100},
#)

# MODIFIED ENVIRONMENTS

gym.envs.register(
    id='MountainCarLong-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=500,
    reward_threshold=-110.0,
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
)

gym.envs.register(
    id='TunnelRAM-v0',
    entry_point='environments:TunnelRAM',
    kwargs={'width': 30, 'height': 30},
)

gym.envs.register(
    id='EvasionWalls-v0',
    entry_point='environments:EvasionWalls',
    kwargs={'width': 30, 'height': 30},
)
