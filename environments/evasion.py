import random

import gym
import matplotlib.pyplot as plt
import numpy

from environments import *


class Evasion(gym.Env):
    AGENT_PIXEL = 0.3
    OBSTACLE_PIXEL = 0.6

    def __init__(self, width: int = 30, height: int = 30, obstacle_chance: float = 0.05):
        self.width = width
        self.height = height
        self.action_space = gym.spaces.Discrete(3)  # UP, DOWN, STRAIGHT
        self.observation_space = gym.spaces.Box(low=0, high=1, dtype=numpy.float16, shape=(self.width, self.height))

        self.dungeon, self.pos_agent = self._init_dungeon()
        # 0.1 = 10% chance that an obstacle is spawned
        self.obstacle_chance_vector = numpy.empty((self.height, 1))
        self.obstacle_chance_vector.fill(obstacle_chance)

        self.max_steps = 500
        self.steps = 0

    def _init_dungeon(self):
        track = numpy.zeros((self.height, self.width))
        pos = self.height // 2
        track[pos, 0] = Evasion.AGENT_PIXEL
        return track, pos

    def reset(self):
        self.steps = 0
        self.dungeon, self.pos_agent = self._init_dungeon()
        return self.dungeon

    def step(self, action):
        self.steps += 1
        done = self.steps >= self.max_steps
        # Straight does not change position of agent
        if action == 0:  # UP
            self.pos_agent = max(0, self.pos_agent - 1)
        elif action == 1:  # DOWN
            self.pos_agent = min(self.height - 1, self.pos_agent + 1)

        # if agent crashed into obstacle --> over
        if self.dungeon[self.pos_agent, 1] != 0:
            done = True
            reward = 0
        else:
            reward = 10

        new_row = numpy.zeros((self.height, 1))
        new_row[numpy.random.binomial(1, self.obstacle_chance_vector) != 0] = Evasion.OBSTACLE_PIXEL
        self.dungeon = numpy.concatenate((self.dungeon[:, 1:], new_row), 1)
        self.dungeon[self.pos_agent, 0] = Evasion.AGENT_PIXEL

        print()
        print(self.dungeon)
        return self.dungeon, reward, done, None

    def render(self, mode='human', close=False):
        img = self.dungeon
        plt.clf()
        plt.imshow(img, cmap="binary", origin="upper")
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.draw()
        plt.pause(0.001)


if __name__ == "__main__":
    env = gym.make('Evasion-v0')

    done = False
    while not done:
        env.render()
        observation, reward, done, _ = env.step(random.choice((2,)))
