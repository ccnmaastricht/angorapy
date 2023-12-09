import mujoco
from matplotlib import pyplot as plt

from angorapy import make_task
from mujoco import viewer
from PIL import Image

from angorapy.common.const import VISION_WH

env = make_task("TestCaseManipulateBlock-v0", render_mode="rgb_array")
env.warmup()

model = env.unwrapped.model
data = env.unwrapped.data
renderer = mujoco.Renderer(model, height=VISION_WH, width=VISION_WH)
cam = env.unwrapped._get_viewer("rgb_array").cam
# cam.distance = 0.3

# open in mujoco viewer
viewer.launch(env.unwrapped.model)


object = data.jnt("block/object:joint/")
quat = data.jnt("block/object:joint/").qpos[3:]
original_qpos = object.qpos.copy()

for rotation in env.unwrapped.test_cases_block_rotations[:6]:
    object.qpos[3:] = rotation
    mujoco.mj_forward(model, data)

    renderer.update_scene(data, camera=cam)
    plt.imshow(renderer.render().copy())
    plt.show()
