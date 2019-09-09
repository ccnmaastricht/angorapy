import math
import os

from learn.exploration import EpsilonGreedyExplorer
from learn.eval import evaluate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import statistics

import gym
import numpy

from datatypes import Experience
from agent.q_learning import DeepQLearningAgent, LinearQLearningAgent
import environments

# SETTINGS
total_iterations = 100000
batch_size = 64

print_every = 1000
avg_over = 20

env = gym.make("CartPole-v0")

number_of_actions = env.action_space.n
state_dimensionality = env.observation_space.shape[0]

agent = DeepQLearningAgent(state_dimensionality, number_of_actions)
explorer = EpsilonGreedyExplorer(eps_decay=0.9999)

episode_rewards = [0]

state = numpy.reshape(env.reset(), [1, -1])
for iteration in range(total_iterations):
    # choose an action and make a step in the environment, based on the agents current knowledge
    action = agent.act(numpy.array(state), explorer=explorer)
    observation, reward, done, info = env.step(action)
    episode_rewards[-1] += reward

    reward = reward if not done else -10
    observation = numpy.reshape(observation, [1, -1])

    # remember experience made during the step in the agents memory
    agent.remember(Experience(state, action, reward, observation), done)

    if iteration % print_every == 0:
        print(f"Iteration {iteration:10d}/{total_iterations} [{round(iteration/total_iterations * 100, 0)}%]"
              f" | {len(episode_rewards) - 1:4d} episodes done"
              f" | last {min(avg_over, len(episode_rewards)):2d}: "
              f"mean {0 if len(episode_rewards) <= 1 else statistics.mean(episode_rewards[-avg_over:-1]):5.2f}; "
              f"max {0 if len(episode_rewards) <= 1 else max(episode_rewards[-avg_over:-1]):5.2f}; "
              f"min {0 if len(episode_rewards) <= 1 else min(episode_rewards[-avg_over:-1]):5.2f}"
              f" | epsilon: {explorer.epsilon}")

    if done:
        state = env.reset()
        episode_rewards.append(0)

    # learn from memory
    if len(agent.memory) > batch_size:
        agent.learn(batch_size)
        explorer.update()

    # go to next state
    state = observation


env.close()