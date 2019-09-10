import os

import gym
import tensorflow as tf

from agent.policy_gradient import REINFORCEAgent

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# activate eager execution to get rid of bullshit static graphs
tf.compat.v1.enable_eager_execution()
tf.keras.backend.set_floatx("float64")  # prevent precision issues

# SETTINGS
TOTAL_EPISODES = 100000

# ENVIRONMENT
env = gym.make("LunarLander-v2")
number_of_actions = env.action_space.n
state_dimensionality = env.observation_space.shape[0]

agent = REINFORCEAgent(state_dimensionality, number_of_actions)
agent.drill(env, TOTAL_EPISODES)

env.close()
