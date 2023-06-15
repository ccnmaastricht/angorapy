import os
from typing import Tuple, Dict

FINGERTIP_SITE_NAMES = [
    'robot/S_fftip',
    'robot/S_mftip',
    'robot/S_rftip',
    'robot/S_lftip',
    'robot/S_thtip',
]

DEFAULT_INITIAL_QPOS = {
    'robot/rh_WRJ2': -0.16514339750464327,
    'robot/rh_WRJ1': -0.31973286565062153,
    'robot/rh_FFJ4': 0.14340512546557435,
    'robot/rh_FFJ3': 0.32028208333591573,
    'robot/rh_FFJ2': 0.7126053607727917,
    'robot/rh_FFJ1': 0.6705281001412586,
    'robot/rh_MFJ4': 0.000246444303701037,
    'robot/rh_MFJ3': 0.3152655251085491,
    'robot/rh_MFJ2': 0.7659800313729842,
    'robot/rh_MFJ1': 0.7323156897425923,
    'robot/rh_RFJ4': 0.00038520700007378114,
    'robot/rh_RFJ3': 0.36743546201985233,
    'robot/rh_RFJ2': 0.7119514095008576,
    'robot/rh_RFJ1': 0.6699446327514138,
    'robot/rh_LFJ5': 0.0525442258033891,
    'robot/rh_LFJ4': -0.13615534724474673,
    'robot/rh_LFJ3': 0.39872030433433003,
    'robot/rh_LFJ2': 0.7415570009679252,
    'robot/rh_LFJ1': 0.704096378652974,
    'robot/rh_THJ5': 0.003673823825070126,
    'robot/rh_THJ4': 0.5506291436028695,
    'robot/rh_THJ3': -0.014515151997119306,
    'robot/rh_THJ2': -0.0015229223564485414,
    'robot/rh_THJ1': -0.7894883021600622,
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

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'mujoco_model/', 'right_hand.xml')
MODEL_PATH_MANIPULATE = os.path.join(os.path.dirname(__file__), '../assets/hand/', 'shadowhand_manipulate.xml')
