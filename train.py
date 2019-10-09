import multiprocessing
import os
import statistics

import numpy
import tensorflow as tf
from gym.spaces import Box

from agent.ppo import PPOAgent
from environments import *
from policy_networks.fully_connected import PPOActorNetwork, PPOCriticNetwork
from utilities.util import env_extract_dims
from utilities.visualization.story import StoryTeller

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# SETTINGS
DEBUG = False
GPU = False

TASK = "LunarLanderContinuous-v2"

ITERATIONS = 1000
WORKERS = 8
HORIZON = 1024 if not DEBUG else 128
EPOCHS = 3
BATCH_SIZE = 64

LEARNING_RATE_POLICY = 3e-4
LEARNING_RATE_CRITIC = 1e-3
DISCOUNT_FACTOR = 0.99
GAE_LAMBDA = 0.97
EPSILON_CLIP = 0.2
C_ENTROPY = 0.01

# setup environment and extract and report information
env = gym.make(TASK)
state_dimensionality, number_of_actions = env_extract_dims(env)
env_action_space_type = "continuous" if isinstance(env.action_space, Box) else "discrete"
env_observation_space_type = "continuous" if isinstance(env.observation_space, Box) else "discrete"
env_name = env.unwrapped.spec.id
print(f"-----------------------------------------\n"
      f"Learning the Task: {env_name}\n"
      f"{state_dimensionality}-dimensional states ({env_observation_space_type}) "
      f"and {number_of_actions} actions ({env_action_space_type}).\n"
      f"-----------------------------------------\n")
if not GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
pool = multiprocessing.Pool(multiprocessing.cpu_count())

# policy and critics networks
policy = PPOActorNetwork(env)
critic = PPOCriticNetwork(env)

# set computation graph to allow for saving
example_input = env.reset().reshape([1, -1]).astype(numpy.float32)
policy.predict(example_input)
critic.predict(example_input)

# set up the agent and a reporting module
agent = PPOAgent(policy, critic, env,
                 horizon=HORIZON,
                 workers=WORKERS,
                 learning_rate_pi=LEARNING_RATE_POLICY,
                 learning_rate_v=LEARNING_RATE_CRITIC,
                 discount=DISCOUNT_FACTOR,
                 epsilon_clip=EPSILON_CLIP,
                 c_entropy=C_ENTROPY,
                 lam=GAE_LAMBDA)
agent.set_gpu(GPU)
teller = StoryTeller(agent, env, frequency=5)

# train
agent.drill(iterations=ITERATIONS,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            worker_pool=pool,
            story_teller=teller)

evaluation_results = agent.evaluate(env, 20, render=False)
print(f"Average Performance {statistics.mean(evaluation_results)}: {evaluation_results}.")

env.close()
