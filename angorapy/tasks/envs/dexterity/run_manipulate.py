from os import wait

import mujoco
from matplotlib import pyplot as plt

from angorapy import make_task
from mujoco import viewer
from PIL import Image

env = make_task("ManipulateBlockDiscreteAsynchronous-v0", render_mode="human")
env.world.robot.show_palm_site()
env.set_delta_t_simulation(0.002)
env.set_original_n_substeps_to_sspcs()
env.change_color_scheme("default")

# viewer.launch(mujoco.MjModel.from_xml_string(env.world.stage.mjcf_model.to_xml_string(), assets=env.world.stage.mjcf_model.get_assets()))
# viewer.launch(env.model, env.data)

while True:
    env.reset()
    done = False
    step = 0

    while not done:
        o, r, t, t2, i = env.step(env.action_space.sample())
        done = t or t2
        # if step % 10 == 0:
        #     plt.imshow(o.vision)
        #     plt.show()

        step += 1