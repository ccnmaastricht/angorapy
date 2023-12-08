from os import wait

import mujoco
from matplotlib import pyplot as plt

from angorapy import make_task
from mujoco import viewer
from PIL import Image

from angorapy.common.const import VISION_WH

env = make_task("TestCaseManipulateBlock-v0", render_mode="human")
env.world.robot.show_palm_site()
env.set_delta_t_simulation(0.002)
env.set_original_n_substeps_to_sspcs()
env.change_color_scheme("default")


model = env.unwrapped.model
data = env.unwrapped.data
renderer = mujoco.Renderer(model, height=VISION_WH, width=VISION_WH)
cam = env.unwrapped._get_viewer("rgb_array").cam

object = data.jnt("block/object:joint/")
quat = data.jnt("block/object:joint/").qpos[3:]
original_qpos = object.qpos.copy()

for rotation in env.unwrapped.test_cases_block_rotations:
    object.qpos[3:] = rotation
    mujoco.mj_forward(model, data)

    renderer.update_scene(data, camera=cam)
    plt.imshow(renderer.render().copy())
    plt.show()