#!/usr/bin/env python
"""Constants used throughout the repository."""
import numpy as np

# MISC
from matplotlib.colors import ListedColormap
import seaborn as sns

COLORS = dict(
    HEADER='\033[95m',
    OKBLUE='\033[94m',
    OKGREEN='\033[92m',
    WARNING='\033[93m',
    FAIL='\033[91m',
    ENDC='\033[0m',
    BOLD='\033[1m',
    UNDERLINE='\033[4m',
    GREEN='\u001b[32m',
    ORANGE="\033[33m"
)

QUALITATIVE_COLOR_PALETTE = [
    "#1D6996",
    "#CC503E",
    "#73AF48",
    "#0F8554",
    "#E17C05",
    "#38A6A5",
    "#EDAD08",
    "#94346E",
    "#5F4690",
    "#6F4070",
    "#994E95",
    "#666666"
]

# MISC
MIN_STAT_EPS = 5
N_SUBSTEPS = 10
SHADOWHAND_MAX_STEPS = 100
SHADOWHAND_SEQUENCE_MAX_STEPS = 100

# STORAGE
BASE_SAVE_PATH = "storage/saved_models/"
STORAGE_DIR = "storage/experience/"
PRETRAINED_COMPONENTS_PATH = "storage/pretraining/"
PATH_TO_EXPERIMENTS = "storage/experiments/"
PATH_TO_BENCHMARKS = "docs/benchmarks/"

# NUMERICAL PRECISION
NP_FLOAT_PREC = np.float64
NUMPY_INTEGER_PRECISION = np.int64
EPSILON = 1e-6  # dont make this lower! 1e-8 would be ignored due to float32 precision

# SHAPES
VISION_WH = 128

# DEBUGGING
DETERMINISTIC = False
DEBUG = False

# OTHER STUFF
COLORMAP = list(ListedColormap(sns.color_palette("deep")).colors)
COLORMAP_HEX = ['#%02x%02x%02x' % tuple(int(c * 255) for c in color) for color in COLORMAP]
