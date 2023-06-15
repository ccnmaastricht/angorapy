import mujoco

from angorapy import make_env
from mujoco import viewer

env = make_env("HumanoidManipulateBlockDiscrete-v0", render_mode="human")
env.world.robot.show_palm_site()

viewer.launch(mujoco.MjModel.from_xml_string(env.world.stage.mjcf_model.to_xml_string(), assets=env.world.stage.mjcf_model.get_assets()))

env.reset()
done = False
while not done:
    o, r, t, t2, i = env.step(env.action_space.sample())
    done = t or t2
