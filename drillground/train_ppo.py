import os
import statistics

import tensorflow as tf
from gym.spaces import Box

from agent.gather import EpisodicGatherer, ContinuousGatherer
from agent.ppo import PPOAgentDual
from environments import *
from policy_networks.fully_connected import PPOActorNetwork, PPOCriticNetwork
from util import env_extract_dims
from visualization.story import StoryTeller

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

# INITIALIZATION
tf.compat.v1.enable_eager_execution()
tf.keras.backend.set_floatx("float64")  # prevent precision issues

# SETTINGS

DEBUG = False

TASK = "Pendulum-v0"  # the environment in which the agent learns
JOINT_NETWORK = False  # if true, uses one network with two heads for policy and critic
GATHERING = ["epi", "cont"][1]  # epi runs n episodes until termination, cont collects specific number of experiences

ITERATIONS = 1000
AGENTS = 32
HORIZON = 4096 if not DEBUG else 128
EPOCHS = 6
BATCH_SIZE = 32

LEARNING_RATE = 0.005
DISCOUNT_FACTOR = 0.99
EPSILON_CLIP = 0.2
C_ENTROPY = 0.01

# SETUP
env = gym.make(TASK)

state_dimensionality, number_of_actions = env_extract_dims(env)
env_action_space_type = "continuous" if isinstance(env.action_space, Box) else "discrete"
env_observation_space_type = "continuous" if isinstance(env.observation_space, Box) else "discrete"
env_name = env.unwrapped.spec.id
print(f"Learning the Task: {env_name}\n"
      f"{state_dimensionality}-dimensional states ({env_observation_space_type}) "
      f"and {number_of_actions} actions ({env_action_space_type}).\n"
      f"-----------------------------------------\n")

gatherer = ContinuousGatherer(environment=env, horizon=HORIZON) if GATHERING == "cont" \
    else EpisodicGatherer(environment=env, n_trajectories=AGENTS)

policy = PPOActorNetwork(env, continuous=env_action_space_type == "continuous")
critic = PPOCriticNetwork(env)
agent = PPOAgentDual(policy, critic, gatherer,
                     learning_rate=LEARNING_RATE,
                     discount=DISCOUNT_FACTOR,
                     epsilon_clip=EPSILON_CLIP,
                     c_entropy=C_ENTROPY)
agent.set_gpu(True)

teller = StoryTeller(agent, env, frequency=10)

# TRAIN
agent.drill(env=env,
            iterations=ITERATIONS,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            story_teller=teller)

evaluation_results = agent.evaluate(env, 10, render=True)
print(f"Average Performance {statistics.mean(evaluation_results)}: {evaluation_results}.")

env.close()
