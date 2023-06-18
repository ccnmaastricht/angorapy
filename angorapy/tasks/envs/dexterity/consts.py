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
    'robot/rh_WRJ2': 0.013937,
    'robot/rh_WRJ1': -0.081399,
    'robot/rh_FFJ4': -0.104522,
    'robot/rh_FFJ3': 0.372759,
    'robot/rh_FFJ2': 0.00979172,
    'robot/rh_FFJ1': 0.474839,
    'robot/rh_MFJ4': -0.0278703,
    'robot/rh_MFJ3': 0.271193,
    'robot/rh_MFJ2': 0.00978675,
    'robot/rh_MFJ1': 0.443133,
    'robot/rh_RFJ4': -0.0522567,
    'robot/rh_RFJ3': 0.262009,
    'robot/rh_RFJ2': 0.00978642,
    'robot/rh_RFJ1': 0.474599,
    'robot/rh_LFJ5': 0.159693,
    'robot/rh_LFJ4': -0.153853,
    'robot/rh_LFJ3': 0.189652,
    'robot/rh_LFJ2': 0.00978933,
    'robot/rh_LFJ1': 0.868464,
    'robot/rh_THJ5': -0.0888116,
    'robot/rh_THJ4': 0.250078,
    'robot/rh_THJ3': -0.050119,
    'robot/rh_THJ2': 0.55893,
    'robot/rh_THJ1': 0.4899,
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
