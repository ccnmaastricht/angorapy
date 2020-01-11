#!/usr/bin/env python
"""Constants."""
import numpy as np

EPS = 1e-8

COLORS = dict(
    HEADER='\033[95m',
    OKBLUE='\033[94m',
    OKGREEN='\033[92m',
    WARNING='\033[93m',
    FAIL='\033[91m',
    ENDC='\033[0m',
    BOLD='\033[1m',
    UNDERLINE='\033[4m'
)

BASE_SAVE_PATH = "storage/saved_models/states/"
STORAGE_DIR = "storage/experience/"
PRETRAINED_COMPONENTS_PATH = "storage/pretrained/"
NUMPY_FLOAT_PRECISION = np.float64
NUMPY_INTEGER_PRECISION = np.int64

# DEBUGGING
DETERMINISTIC = False
DEBUG = False
