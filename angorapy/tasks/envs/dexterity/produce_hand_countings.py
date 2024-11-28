import time

import mujoco
from matplotlib import pyplot as plt

from angorapy import make_task
from mujoco import viewer
from PIL import Image

from angorapy.common.const import VISION_WH
import numpy as np

COUNTING_ONE = '-0.0707368 -0.169821 0.142963 0.295422 0.22301 0.180817 0.0279382 1.56138 1.49844 1.56234 0.000389537 1.56137 1.49844 1.56234 0.00979388 -0.136212 1.56138 1.49845 1.56234 0.302336 0.422192 0.199217 -0.514455 -1.56137 0 0 -12510.1 1 0 0 0'
COUNTING_TWO = '-0.0707368 -0.173724 0.142955 0.29548 0.223074 0.273577 0.000249447 0.290421 0.222933 0.256321 0.000389594 1.56137 1.49844 1.56234 0.0097937 -0.136214 1.56138 1.49845 1.56234 0.302352 0.42219 0.199217 -0.514455 -1.56137 0 0 -9728.97 1 0 0 0'
COUNTING_THREE = '-0.0707368 -0.177513 0.142948 0.2955 0.2231 0.310264 0.000249928 0.290444 0.222961 0.296904 0.00038816 0.342696 0.207318 0.281257 0.00979355 -0.136212 1.56138 1.49845 1.56234 0.302364 0.422194 0.199217 -0.514455 -1.56137 0 0 -8338.4 1 0 0 0'
COUNTING_FOUR = '-0.0707368 -0.181201 0.142941 0.295534 0.223138 0.359826 0.000249913 0.290481 0.223002 0.351941 0.000388138 0.342727 0.207355 0.330767 0.0095106 -0.135595 0.374933 0.223226 0.350424 0.302376 0.422196 0.199217 -0.514455 -1.56137 0 0 -6000.02 1 0 0 0'
COUNTING_FIVE = '-0.0707223 -0.181512 0.142934 0.295328 0.221419 0.110848 0.0278316 0.209656 0.0968418 0.0831339 0.000388255 0.233249 0.104712 0.0472694 0.00950084 -0.135549 0.209751 0.136202 0.172461 -0.19658 0.140207 0.0197734 -0.156674 -0.439795 0 0 -17455 1 0 0 0'

env = make_task("TestCaseManipulateBlock-v0", render_mode="rgb_array")
env.reset()
env.calc_rotation_set()

model = env.unwrapped.model
data = env.unwrapped.data
renderer = mujoco.Renderer(model, height=1024, width=512)
cam = env.unwrapped._get_viewer("rgb_array").cam
cam.lookat[0] = 0.4

object = data.jnt("block/object:joint/")
quat = data.jnt("block/object:joint/").qpos[3:]
original_qpos = object.qpos.copy()

for i, count in enumerate([COUNTING_ONE, COUNTING_TWO, COUNTING_THREE, COUNTING_FOUR, COUNTING_FIVE], start=1):
    data.qpos = np.array([float(s) for s in count.split(" ")])
    # remove the object from the scene
    object.qpos = np.array([0, 0, 0, 0, 0, 0, 0])

    mujoco.mj_forward(model, data)

    renderer.update_scene(data, camera=cam)
    plt.imsave(f"counting_{i}.png", renderer.render().copy())
