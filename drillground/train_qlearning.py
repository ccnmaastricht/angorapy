#!/usr/bin/env python
import os
import gym
from agent.q_learning import DeepQLearningAgent
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# SETTINGS
TOTAL_ITERATIONS = 100000
batch_size = 64

print_every = 1000
avg_over = 20

env = gym.make("CartPole-v0")

number_of_actions = env.action_space.n
state_dimensionality = env.observation_space.shape[0]

agent = DeepQLearningAgent(state_dimensionality, number_of_actions)
agent.drill(env, TOTAL_ITERATIONS)

env.close()
