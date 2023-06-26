import os
from typing import Tuple, Dict

FINGERTIP_SITE_NAMES = [
    'robot/rh_S_fftip',
    'robot/rh_S_mftip',
    'robot/rh_S_rftip',
    'robot/rh_S_lftip',
    'robot/rh_S_thtip',
]

# DEFAULT_INITIAL_QPOS = {
#     'robot/rh_WRJ2': 0.013937,
#     'robot/rh_WRJ1': -0.081399,
#     'robot/rh_FFJ4': -0.104522,
#     'robot/rh_FFJ3': 0.372759,
#     'robot/rh_FFJ2': 0.00979172,
#     'robot/rh_FFJ1': 0.474839,
#     'robot/rh_MFJ4': -0.0278703,
#     'robot/rh_MFJ3': 0.271193,
#     'robot/rh_MFJ2': 0.00978675,
#     'robot/rh_MFJ1': 0.443133,
#     'robot/rh_RFJ4': -0.0522567,
#     'robot/rh_RFJ3': 0.262009,
#     'robot/rh_RFJ2': 0.00978642,
#     'robot/rh_RFJ1': 0.474599,
#     'robot/rh_LFJ5': 0.159693,
#     'robot/rh_LFJ4': -0.153853,
#     'robot/rh_LFJ3': 0.189652,
#     'robot/rh_LFJ2': 0.00978933,
#     'robot/rh_LFJ1': 0.868464,
#     'robot/rh_THJ5': -0.0888116,
#     'robot/rh_THJ4': 0.250078,
#     'robot/rh_THJ3': -0.050119,
#     'robot/rh_THJ2': 0.55893,
#     'robot/rh_THJ1': 0.4899,
# }

DEFAULT_INITIAL_QPOS = {
    'robot/rh_WRJ1': -0.16514339750464327,
    'robot/rh_WRJ0': -0.31973286565062153,
    'robot/rh_FFJ3': 0.14340512546557435,
    'robot/rh_FFJ2': 0.32028208333591573,
    'robot/rh_FFJ1': 0.7126053607727917,
    'robot/rh_FFJ0': 0.6705281001412586,
    'robot/rh_MFJ3': 0.000246444303701037,
    'robot/rh_MFJ2': 0.3152655251085491,
    'robot/rh_MFJ1': 0.7659800313729842,
    'robot/rh_MFJ0': 0.7323156897425923,
    'robot/rh_RFJ3': 0.00038520700007378114,
    'robot/rh_RFJ2': 0.36743546201985233,
    'robot/rh_RFJ1': 0.7119514095008576,
    'robot/rh_RFJ0': 0.6699446327514138,
    'robot/rh_LFJ4': 0.0525442258033891,
    'robot/rh_LFJ3': -0.13615534724474673,
    'robot/rh_LFJ2': 0.39872030433433003,
    'robot/rh_LFJ1': 0.7415570009679252,
    'robot/rh_LFJ0': 0.704096378652974,
    'robot/rh_THJ4': 0.003673823825070126,
    'robot/rh_THJ3': 0.5506291436028695,
    'robot/rh_THJ2': -0.014515151997119306,
    'robot/rh_THJ1': -0.0015229223564485414,
    'robot/rh_THJ0': -0.7894883021600622,
}

FINGERTIP_BODIES: Tuple[str, ...] = (
    "thdistal",
    "ffdistal",
    "mfdistal",
    "rfdistal",
    "lfdistal",
)

JOINT_GROUP: Dict[str, Tuple[str, ...]] = {
    "wrist": ("WRJ1", "WRJ0"),
    "thumb": ("THJ4", "THJ3", "THJ2", "THJ1", "THJ0"),
    "first": ("FFJ3", "FFJ2", "FFJ1", "FFJ0"),
    "middle": ("MFJ3", "MFJ2", "MFJ1", "MFJ0"),
    "ring": ("RFJ3", "RFJ2", "RFJ1", "RFJ0"),
    "little": ("LFJ4", "LFJ3", "LFJ2", "LFJ1", "LFJ0"),
}