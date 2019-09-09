import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import statistics

import gym
import numpy

from datatypes import Experience
from agent.policy_gradient import REINFORCEAgent

import tensorflow as tf

# activate eager execution to get rid of bullshit staticness
tf.compat.v1.enable_eager_execution()
tf.keras.backend.set_floatx("float64")  # prevent precision issues

# SETTINGS
TOTAL_EPISODES = 100000

# ENVIRONMENT
env = gym.make("CartPole-v1")
number_of_actions = env.action_space.n
state_dimensionality = env.observation_space.shape[0]

# AGENT
agent = REINFORCEAgent(state_dimensionality, number_of_actions)

# TRAINING
episode_reward_history = []
for episode in range(TOTAL_EPISODES):
    partial_episode_gradients = []
    reward_trajectory = []

    state = numpy.reshape(env.reset(), [1, -1])
    done = False
    while not done:
        if episode % 100 == 0:
            env.render()

        # choose action and calculate partial loss (not yet weighted on future reward)
        with tf.GradientTape() as tape:
            action, action_probability = agent.act(state)
            partial_loss = agent.loss(action_probability)

        # get and remember unweighted gradient
        partial_episode_gradients.append(tape.gradient(partial_loss, agent.model.trainable_variables))

        # actually apply the chosen action
        observation, reward, done, _ = env.step(action)
        observation = numpy.reshape(observation, [1, -1])
        reward_trajectory.append(reward)

        if not done:
            # next state is observation after executing the action
            state = observation

    # gather future rewards and apply them to partial gradients
    discounted_future_rewards = [agent.discount**t * sum(reward_trajectory[t:]) for t in range(len(reward_trajectory))]
    full_gradients = [
        [gradient_tensor * discounted_future_rewards[t]
         for gradient_tensor in partial_episode_gradients[t]] for t in range(len(discounted_future_rewards))
    ]

    # optimize the policy based on all time steps
    for t in range(len(full_gradients)):
        agent.optimizer.apply_gradients(zip(full_gradients[t], agent.model.trainable_variables))

    episode_reward_history.append(sum(reward_trajectory))

    if episode % 30 == 0 and len(episode_reward_history) > 0:
        print(f"Episode {episode:10d}/{TOTAL_EPISODES} | Mean over last 30: {statistics.mean(episode_reward_history[-30:]):4.2f}")

env.close()