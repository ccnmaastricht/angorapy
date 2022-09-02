import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

from angorapy.common.wrappers import make_env

import matplotlib.pyplot as plt

tf.get_logger().setLevel('INFO')

env = make_env("HumanoidVisualManipulateBlock-v0")

state = env.reset()

for i in range(100):
    state, r, dd, info = env.step(env.action_space.sample())

plt.imshow(state.vision / 255)
plt.show()
