import os

import gym
import tensorflow as tf

from agent.ppo import PPOAgent, PPOAgentDual
from configs.env import CONFIG

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# activate eager execution to get rid of bullshit static graphs
tf.compat.v1.enable_eager_execution()
tf.keras.backend.set_floatx("float64")  # prevent precision issues

# ENVIRONMENT
# env = gym.make("LunarLander-v2")
env = gym.make("CartPole-v0")
number_of_actions = env.action_space.n
state_dimensionality = env.observation_space.shape[0]
env_name = env.spec._env_name

print(env_name)
print(f"{state_dimensionality}-dimensional states and {number_of_actions} actions.")

# AGENT
agent = PPOAgentDual(state_dimensionality,
                 number_of_actions,
                 learning_rate=CONFIG["PPO"][env_name]["BEST"]["LEARNING_RATE"],
                 discount=CONFIG["PPO"][env_name]["BEST"]["DISCOUNT_FACTOR"],
                 epsilon_clip=CONFIG["PPO"][env_name]["BEST"]["EPSILON_CLIP"])

agent.set_gpu(False)
agent.drill(env=env,
            iterations=CONFIG["PPO"][env_name]["BEST"]["ITERATIONS"],
            agents=CONFIG["PPO"][env_name]["BEST"]["AGENTS"],
            epochs=CONFIG["PPO"][env_name]["BEST"]["EPOCHS"],
            batch_size=CONFIG["PPO"][env_name]["BEST"]["BATCH_SIZE"])

env.close()
