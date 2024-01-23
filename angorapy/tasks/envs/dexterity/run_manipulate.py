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

# env.set_chain_code("l_r_u_d_c_a")

model = env.model
data = env.data
renderer = mujoco.Renderer(model, height=VISION_WH, width=VISION_WH)
cam = env.unwrapped._get_viewer("rgb_array").cam
cam.distance = 0.5  # zoom in
cam.azimuth = 35.0  # wrist to the bottom
cam.elevation = -125.0  # top down view
cam.lookat[0] += 0.  # slightly move forward
cam.lookat[1] += 0.0  # slightly move forward

# show interactive viewer
# viewer.launch(model, data)

object = data.jnt("block/object:joint/")
quat = data.jnt("block/object:joint/").qpos[3:]
original_qpos = object.qpos.copy()

for rotation in [original_qpos[3:], *env.target_chain]:
    object.qpos[3:] = rotation
    mujoco.mj_forward(model, data)

    renderer.update_scene(data, camera=cam)
    plt.imshow(renderer.render().copy())
    plt.show()
