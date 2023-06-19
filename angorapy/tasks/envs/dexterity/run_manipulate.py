from os import wait

import mujoco
from matplotlib import pyplot as plt

from angorapy import make_env
from mujoco import viewer
from PIL import Image

env = make_env("HumanoidManipulateBlockDiscreteAsynchronous-v0", render_mode="human")
env.world.robot.show_palm_site()

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

        print(f"Step {step}: {env.data.site('robot/palm_center_site').xpos} vs {env.data.site('block/object:center').xpos}")
        if done:
            print("ITS OVER!")
            wait()

        step += 1