import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

from angorapy import make_task

import matplotlib.pyplot as plt

tf.get_logger().setLevel('INFO')

env = make_task("ManipulateBlockVisualDiscrete-v0", render_mode="rgb_array")

state = env.reset()

for i in range(100):
    state, r, dd, truncated, info = env.step(env.action_space.sample())

plt.imshow(state.vision / 1)
plt.show()
