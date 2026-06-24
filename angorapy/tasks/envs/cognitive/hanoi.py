"""Implementation from https://github.com/RobertTLange/gym-hanoi/blob/master/gym_hanoi/envs/hanoi_env.py"""

import itertools
import random
from typing import Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from angorapy.common.senses import Sensation
from angorapy.tasks.utils import convert_observation_to_space


class HanoiEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, render_mode=None):
        self.num_disks = 3
        self.env_noise = 0

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        # pygame rendering handles (lazily initialised on first render call)
        self.window = None
        self.clock = None

        self.current_state: Union[None, np.ndarray] = None
        self.goal_state: Union[None, np.ndarray] = np.array(self.num_disks * (2,))

        self.step_count = 0
        self.done = None
        self.ACTION_LOOKUP = {0: "(0,1) - top disk of pole 0 to top of pole 1 ",
                              1: "(0,2) - top disk of pole 0 to top of pole 2 ",
                              2: "(1,0) - top disk of pole 1 to top of pole 0",
                              3: "(1,2) - top disk of pole 1 to top of pole 2",
                              4: "(2,0) - top disk of pole 2 to top of pole 0",
                              5: "(2,1) - top disk of pole 2 to top of pole 1"}

        self.action_space = spaces.Discrete(6)
        self.observation_space = convert_observation_to_space(self.reset()[0])

    def step(self, action):
        """
        * Inputs:
            - action: integer from 0 to 5 (see ACTION_LOOKUP)
        * Outputs:
            - current_state: state after transition
            - reward: reward from transition
            - done: episode state
            - info: dict of booleans (noisy?/invalid action?)
        0. Check if transition is noisy or not
        1. Transform action (0 to 5 integer) to tuple move - see Lookup
        2. Check if move is allowed
        3. If it is change corresponding entry | If not return same state
        4. Check if episode completed and return
        """
        if self.done:
            raise RuntimeError("Episode has finished. Call env.reset() to start a new episode.")

        self.step_count += 1

        info = {"transition_failure": False,
                "invalid_action": False}

        if self.env_noise > 0:
            r_num = random.random()
            if r_num <= self.env_noise:
                action = random.randint(0, self.action_space.n - 1)
                info["transition_failure"] = True

        move = action_to_move[action]

        if self.move_allowed(move):
            disk_to_move = min(self.disks_on_peg(move[0]))
            moved_state = list(self.current_state)
            moved_state[disk_to_move] = move[1]
            self.current_state = np.array(tuple(moved_state))
        else:
            info["invalid_action"] = True

        if np.all(self.current_state == self.goal_state):
            reward = 100
            self.done = True
        elif info["invalid_action"] == True:
            reward = -1
        else:
            reward = -0.01

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, self.done, self.done, info

    def _get_obs(self):
        return {
            "observation": Sensation(
                proprioception=np.array(self.current_state),
            ),
        }

    def disks_on_peg(self, peg):
        """
        * Inputs:
            - peg: pole to check how many/which disks are in it
        * Outputs:
            - list of disk numbers that are allocated on pole
        """
        return [disk for disk in range(self.num_disks) if self.current_state[disk] == peg]

    def move_allowed(self, move):
        """
        * Inputs:
            - move: tuple of state transition (see ACTION_LOOKUP)
        * Outputs:
            - boolean indicating whether action is allowed from state!
        move[0] - peg from which we want to move disc
        move[1] - peg we want to move disc to
        Allowed if:
            * discs_to is empty (no disc of peg) set to true
            * Smallest disc on target pole larger than smallest on prev
        """
        disks_from = self.disks_on_peg(move[0])
        disks_to = self.disks_on_peg(move[1])

        if disks_from:
            return (min(disks_to) > min(disks_from)) if disks_to else True
        else:
            return False

    def reset(self, **kwargs):
        self.current_state = np.array(self.num_disks * (0,))
        self.step_count = 0
        self.done = False

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}

    def render(self):
        """Render the current configuration of the towers using pygame.

        Honours ``self.render_mode``: ``"human"`` opens a window and draws to it,
        ``"rgb_array"`` returns an ``(H, W, 3)`` uint8 numpy array instead.
        """
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render() without specifying a render mode. "
                "Set render_mode at construction, e.g. HanoiEnv(render_mode='rgb_array')."
            )
            return None

        try:
            import pygame
        except ImportError as e:
            raise gym.error.DependencyNotInstalled(
                "pygame is not installed, run `pip install pygame` to use the Hanoi renderer."
            ) from e

        width, height = 600, 400
        peg_color = (90, 60, 30)
        base_color = (60, 40, 20)
        bg_color = (245, 245, 245)
        disk_colors = [
            (220, 50, 50), (240, 150, 30), (240, 220, 40),
            (60, 200, 80), (50, 120, 220), (150, 60, 200),
            (120, 120, 120),
        ]

        # lazily set up the display / surface
        if self.window is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Tower of Hanoi")
                self.window = pygame.display.set_mode((width, height))
            else:
                self.window = pygame.Surface((width, height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((width, height))
        canvas.fill(bg_color)

        # geometry of the three pegs and the base they stand on
        base_height = 20
        base_top = height - 40
        peg_height = 220
        peg_width = 10
        peg_xs = [width * (i + 1) // 4 for i in range(3)]

        pygame.draw.rect(canvas, base_color, (40, base_top, width - 80, base_height))
        for peg_x in peg_xs:
            pygame.draw.rect(
                canvas, peg_color,
                (peg_x - peg_width // 2, base_top - peg_height, peg_width, peg_height),
            )

        # draw disks: larger disk index == larger disk, stacked largest at bottom
        disk_height = 22
        min_disk_width = 36
        max_disk_width = (width // 4) - 20
        if self.num_disks > 1:
            width_step = (max_disk_width - min_disk_width) / (self.num_disks - 1)
        else:
            width_step = 0

        for peg in range(3):
            disks = sorted(self.disks_on_peg(peg), reverse=True)  # largest first (bottom)
            for level, disk in enumerate(disks):
                disk_width = int(min_disk_width + width_step * disk)
                disk_x = peg_xs[peg] - disk_width // 2
                disk_y = base_top - (level + 1) * disk_height
                pygame.draw.rect(
                    canvas, disk_colors[disk % len(disk_colors)],
                    (disk_x, disk_y, disk_width, disk_height - 2),
                    border_radius=4,
                )

        # step counter in the top-right corner
        font = pygame.font.SysFont(None, 28)
        label = font.render(f"Step: {self.step_count}", True, (40, 40, 40))
        canvas.blit(label, (width - label.get_width() - 12, 12))

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])
            return None
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            import pygame
            if self.render_mode == "human":
                pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None

    def get_movability_map(self, fill=False):
        # Initialize movability map
        mov_map = np.zeros(self.num_disks * (3,) + (6,))

        if fill:
            # Get list of all states as tuples
            id_list = self.num_disks * [0] + self.num_disks * [1] + self.num_disks * [2]
            states = list(itertools.permutations(id_list, self.num_disks))

            for state in states:
                for action in range(6):
                    move = action_to_move[action]
                    disks_from = []
                    disks_to = []
                    for d in range(self.num_disks):
                        if state[d] == move[0]:
                            disks_from.append(d)
                        elif state[d] == move[1]:
                            disks_to.append(d)

                    if disks_from:
                        valid = (min(disks_to) > min(disks_from)) if disks_to else True
                    else:
                        valid = False

                    if not valid: mov_map[state][action] = -np.inf

                    move_from = [m[0] for m in action_to_move]
                    move_to = [m[1] for m in action_to_move]

        # # Try to get rid of action loop - vectorize...
        # for state in states:
        #     s = np.array(state)
        #     disks_from = []
        #     disks_to = []
        #
        #     for d in range(self.num_disks):
        #         a_from = [a for a, v in enumerate(move_from) if v == s[d]]
        #         a_to = [a for a, v in enumerate(move_to) if v == s[d]]
        #
        #         if disks_from:
        #             valid = (min(disks_to) > min(disks_from)) if disks_to else True
        #         else:
        #             valid = False
        #
        #         if not valid:
        #             mov_map[state][action] = -np.inf
        return mov_map


action_to_move = [(0, 1), (0, 2), (1, 0),
                  (1, 2), (2, 0), (2, 1)]
