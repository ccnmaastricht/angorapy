import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from environments import *

from agent.ppo_agent import PPOAgent
from common.wrappers import make_env
from dexterity.models import get_model_builder

import matplotlib.pyplot as plt

tf.get_logger().setLevel('INFO')

env = make_env("ReachAbsoluteVisual-v0")
agent = PPOAgent(get_model_builder("shadow", "gru"), env, 1024, 8)

state = env.reset()

for i in range(100):
    state, r, dd, info = env.step(env.action_space.sample())

plt.imshow(state.vision / 255)
plt.show()