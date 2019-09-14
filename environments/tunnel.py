import matplotlib.pyplot as plt
import numpy

from environments import *


class Tunnel(gym.Env):
    AGENT_PIXEL = 0.3
    OBSTACLE_PIXEL = 0.6

    def __init__(self, width: int = 30, height: int = 30):
        self.width = width
        self.height = height
        self.action_space = gym.spaces.Discrete(3)  # UP, DOWN, STRAIGHT
        self.observation_space = gym.spaces.Box(low=0, high=1, dtype=numpy.float16, shape=(self.width, self.height))

        self.tunnel_center = numpy.random.randint(low=1, high=self.height, size=(1,)).item()
        self.dungeon, self.pos_agent = self._init_dungeon()
        self.max_steps = 500
        self.steps = 0

    def _init_dungeon(self):
        track = numpy.zeros((self.height, self.width))
        pos = self.tunnel_center
        for row in range(self.width):
            new_row = numpy.empty((self.height, 1))
            new_row.fill(Tunnel.OBSTACLE_PIXEL)

            # create tunnel
            new_row[max(0, self.tunnel_center - 3):min(self.tunnel_center + 3, self.height)] = 0

            # move tunnel up or down
            self.tunnel_center = min(
                max(numpy.array([1], dtype=int), self.tunnel_center + numpy.random.randint(low=-1, high=2, size=(1,))),
                numpy.array([self.height - 1], dtype=int)).item()

            # new_row[torch.bernoulli(self.obstacle_chance).byte()] = Tunnel.OBSTACLE_PIXEL
            track = numpy.concatenate((track[:, 1:], new_row), 1)

            # remember leftmost rows center for agent start position
            if row == 0:
                pos = self.tunnel_center
        track[pos, 0] = Tunnel.AGENT_PIXEL
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

        new_row = numpy.empty((self.height, 1))
        new_row.fill(Tunnel.OBSTACLE_PIXEL)
        # create tunnel
        new_row[max(0, self.tunnel_center - 3):min(self.tunnel_center + 3, self.height)] = 0
        # move tunnel up or down
        self.tunnel_center = min(
            max(numpy.array([1]), self.tunnel_center + numpy.random.randint(low=-1, high=2, size=(1,))),
            numpy.array([self.height - 1])).item()
        # new_row[torch.bernoulli(self.obstacle_chance).byte()] = Tunnel.OBSTACLE_PIXEL
        self.dungeon = numpy.concatenate((self.dungeon[:, 1:], new_row), 1)
        self.dungeon[self.pos_agent, 0] = Tunnel.AGENT_PIXEL

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
    env = gym.make("Tunnel-v0")
    # env = gym.make('Evasion-v0')
    for _ in range(5):
        env.reset()
        done = False
        while not done:
            env.render()
            observation, reward, done, _ = env.step(2)
