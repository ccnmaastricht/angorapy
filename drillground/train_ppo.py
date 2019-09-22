import os

import tensorflow as tf

from agent.gathering import EpisodicGatherer, ContinuousGatherer
from agent.ppo import PPOAgentDual
from configs.env import CONFIG
from environments import *
from policy_networks.fully_connected import PPOActorNetwork, PPOCriticNetwork
from util import env_extract_dims

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

# INITIALIZATION
tf.compat.v1.enable_eager_execution()
tf.keras.backend.set_floatx("float64")  # prevent precision issues

# SETTINGS
TASK = "CartPole-v0"  # the environment in which the agent learns
HP_CONFIG = "BEST"  # configuration of hyper parameters in configs/env.py
JOINT_NETWORK = False  # if true, uses one network with two heads for policy and critic
GATHERING = ["epi", "cont"][0]  # epi runs n episodes until termination, cont collects specific number of experiences

# SETUP
env = gym.make(TASK)

state_dimensionality, number_of_actions = env_extract_dims(env)
env_name = env.spec._env_name
print(f"Learning the Task: {env_name}\n"
      f"{state_dimensionality}-dimensional states and {number_of_actions} actions.\n"
      f"-----------------------------------------\n\n")

if GATHERING == "cont":
    gatherer = ContinuousGatherer(environment=env,
                                  n_trajectories=CONFIG["PPO"][env_name][HP_CONFIG]["AGENTS"],
                                  T=200)
else:
    gatherer = EpisodicGatherer(environment=env,
                                n_trajectories=CONFIG["PPO"][env_name][HP_CONFIG]["AGENTS"])

if not JOINT_NETWORK:
    policy = PPOActorNetwork(env)
    critic = PPOCriticNetwork(env)

    agent = PPOAgentDual(policy, critic, gatherer,
                         learning_rate=CONFIG["PPO"][env_name][HP_CONFIG]["LEARNING_RATE"],
                         discount=CONFIG["PPO"][env_name][HP_CONFIG]["DISCOUNT_FACTOR"],
                         epsilon_clip=CONFIG["PPO"][env_name][HP_CONFIG]["EPSILON_CLIP"])
else:
    raise NotImplementedError("Implemented but currently not used.")
agent.set_gpu(True)

# TRAIN
agent.drill(env=env,
            iterations=CONFIG["PPO"][env_name][HP_CONFIG]["ITERATIONS"],
            epochs=CONFIG["PPO"][env_name][HP_CONFIG]["EPOCHS"],
            batch_size=CONFIG["PPO"][env_name][HP_CONFIG]["BATCH_SIZE"])
print(agent.evaluate(env, 1, render=True))

env.close()
