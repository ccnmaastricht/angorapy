import numpy

from environments import *


class Tunnel(gym.Env):
    """A game environment where the player (hence the agent) is tasked with escaping a tunnel.

    The tunnel is a white trail of height 3 pixels, over a black background. The environment scrolls to the left and
    the tunnel randomly moves upwards or downwards, s.t. the agent has to move in order to not crash into the tunnel.

    There are 4 modes in which the environment can be used, determining the state information.

        - ram:  the arguably easiest mode in which the agent gets the position of the tunnel border above and below and
                its own position.
        - rows: the agent gets the flattened first two columns (slight wording-whoopsie), including itself.
        - flat: the agent gets the entire environment as pixels, flattened to a vector. contains a lot of unnecessary
                information he needs to lear to ignore.
        - None: the dungeon is presented as is, i.e. a 2D image.
    """

    AGENT_PIXEL = 100
    OBSTACLE_PIXEL = 255

    def __init__(self, width: int = 30, height: int = 30, mode: str = None):
        self.mode = mode
        self.width = width
        self.height = height
        self.action_space = gym.spaces.Discrete(3)  # UP, DOWN, STAY

        if mode is None:
            self.observation_space = gym.spaces.Box(low=0, high=1, dtype=numpy.float16, shape=(self.width, self.height))
        elif mode == "flat":
            self.observation_space = gym.spaces.Box(low=0, high=1, dtype=numpy.float16, shape=(self.width * self.height,))
        elif mode == "rows":
            self.observation_space = gym.spaces.Box(low=0, high=1, dtype=numpy.float16, shape=(2 * self.height,))
        elif mode == "ram":
            self.observation_space = gym.spaces.Box(low=0, high=32, shape=(3,))
        else:
            raise ValueError("Unknown Tunnel Mode.")

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
        """Reset the environment and return an initial state."""
        self.steps = 0
        self.dungeon, self.pos_agent = self._init_dungeon()

        return self.make_state_representation()

    def step(self, action):
        """Make a step in the environment."""
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

        return self.make_state_representation(), reward, done, {}

    def make_state_representation(self):
        """Make a state representation dependent on the mode."""
        if self.mode == "flat":
            representation = self.dungeon.reshape([-1])
        elif self.mode == "rows":
            representation = self.dungeon[:, :2].reshape([-1])
        elif self.mode == "ram":
            empty_pixels = numpy.argwhere(self.dungeon[:, 1] != self.OBSTACLE_PIXEL)
            top_boundary, bottom_boundary = numpy.max(empty_pixels), numpy.min(empty_pixels)
            representation = numpy.array([self.pos_agent, top_boundary.item(), bottom_boundary.item()])
        else:
            representation = numpy.expand_dims(self.dungeon, axis=-1)  # add channel dimension

        return representation

    def render(self, mode='human', close=False):
        """Render the dungeon."""
        return self.dungeon


if __name__ == "__main__":
    env = gym.make("TunnelRAM-v0")
    for _ in range(5):
        env.reset()
        done = False
        while not done:
            env.render()
            observation, r, done, _ = env.step(2)
            print(observation)
