from pathlib import Path
from typing import Dict, Tuple


JOINT_GROUP: Dict[str, Tuple[str, ...]] = {
    "wrist": ("WRJ1", "WRJ0"),
    "thumb": ("THJ4", "THJ3", "THJ2", "THJ1", "THJ0"),
    "first": ("FFJ3", "FFJ2", "FFJ1", "FFJ0"),
    "middle": ("MFJ3", "MFJ2", "MFJ1", "MFJ0"),
    "ring": ("RFJ3", "RFJ2", "RFJ1", "RFJ0"),
    "little": ("LFJ4", "LFJ3", "LFJ2", "LFJ1", "LFJ0"),
}

FINGERTIP_BODIES: Tuple[str, ...] = (
    # Important: the order of these names should not be changed.
    "thdistal",
    "ffdistal",
    "mfdistal",
    "rfdistal",
    "lfdistal",
)

FINGERTIP_COLORS: Tuple[Tuple[float, float, float], ...] = (
    # Important: the order of these colors should not be changed.
    (0.8, 0.2, 0.8),  # Purple.
    (0.8, 0.2, 0.2),  # Red.
    (0.2, 0.8, 0.8),  # Cyan.
    (0.2, 0.2, 0.8),  # Blue.
    (0.8, 0.8, 0.2),  # Yellow.
)

MODEL_XML = Path(__file__).resolve().parent / "bodies" / "gym_shadowhand.xml"
