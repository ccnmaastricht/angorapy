import time

import mujoco
from matplotlib import pyplot as plt

from angorapy import make_task
from mujoco import viewer
from PIL import Image

from angorapy.common.const import VISION_WH
import numpy as np


env = make_task("TestCaseManipulateBlockRandomized-v0", render_mode="rgb_array")
env.reset()

# run an episode and print out the current position in the chain
done = False
while not done:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(env.position_in_chain)


# model = env.model
# data = env.data
# renderer = mujoco.Renderer(model, height=VISION_WH, width=VISION_WH)
# cam = env.unwrapped._get_viewer("rgb_array").cam
# cam.distance = 0.42  # zoom
# cam.azimuth = 45  # wrist to the bottom
# cam.elevation = -60.0  # top down view
# cam.lookat[0] = 0.45  # slightly move forward
# cam.lookat[1] = 0  # slightly move left
# cam.lookat[2] = 0  # slightly move left
#
# # show interactive viewer
# # viewer.launch(model, data)
#
# object = data.jnt("block/object:joint/")
# quat = data.jnt("block/object:joint/").qpos[3:]
# original_qpos = object.qpos.copy()
#
# print(env.chain_code)
# for rotation in [original_qpos[3:], *env.target_chain]:
#     object.qpos[3:] = rotation
#     mujoco.mj_forward(model, data)
#
#     renderer.update_scene(data, camera=cam)
#     plt.imshow(renderer.render().copy())
#     plt.show()
