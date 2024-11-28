import time

import mujoco
from matplotlib import pyplot as plt

from angorapy import make_task
from mujoco import viewer
from PIL import Image

from angorapy.common.const import VISION_WH
import numpy as np


env = make_task("TestCaseManipulateBlock-v0", render_mode="rgb_array")
env.reset()
env.calc_rotation_set()

model = env.unwrapped.model
data = env.unwrapped.data
renderer = mujoco.Renderer(model, height=VISION_WH, width=VISION_WH)
cam = env.unwrapped._get_viewer("rgb_array").cam
# cam.distance = 0.3

object = data.jnt("block/object:joint/")
quat = data.jnt("block/object:joint/").qpos[3:]
original_qpos = object.qpos.copy()

COUNTING_ONE = '-0.165094 -0.162238 0.215782 0.295069 0.00952981 0.00993631 0.000250818 1.56138 1.49844 1.56233 -0.0803201 1.56138 1.49844 1.56233 0.123181 -0.199979 1.56128 1.49843 1.56233 0.0597197 0.677516 -0.199459 -0.514419 -1.56137 0.321965 0.0138695 0.0350282 0.538976 0.524173 -0.369853 0.545855'
data.qpos = np.array([float(s) for s in COUNTING_ONE.split(" ")])
# remove the object from the scene
object.qpos = np.array([0, 0, 0, 0, 0, 0, 0])
mujoco.mj_forward(model, data)

# open mujoco viewer and prevent python from closing it
viewer.launch(model, data)
