import mujoco

from angorapy import make_env
from mujoco import viewer

env = make_env("HumanoidManipulateBlockDiscrete-v0", render_mode="human")
# viewer.launch(mujoco.MjModel.from_xml_string(env.env.world.stage.mjcf_model.to_xml_string(), assets=env.env.world.stage.mjcf_model.get_assets()))

env.reset()
done = False
while not done:
    o, r, t, t2, i = env.step(env.action_space.sample())
    done = t or t2
