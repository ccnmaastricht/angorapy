#!/usr/bin/env python
"""Methods for creating a story about a training process."""
import os
import re
import time

import gym
import matplotlib.pyplot as plt
from matplotlib import animation

import tensorflow as tf

from agent.core import RandomAgent, _RLAgent

PATH_TO_STORIES = "../docs/stories/"


class StoryTeller:

    def __init__(self, agent: _RLAgent, env: gym.Env, frequency: int, id=None):
        self.agent = agent
        self.env = env

        self.frequency = frequency

        if id is None:
            self.story_id = round(time.time())
            self.story_directory = f"{PATH_TO_STORIES}/experiments/{self.story_id}/"
            os.makedirs(self.story_directory)
        else:
            self.story_id = id
            self.story_directory = f"{PATH_TO_STORIES}/experiments/{self.story_id}/"
            if not os.path.isdir(self.story_directory):
                raise ValueError("Given ID not found in experiments.")

        self.make_metadata()

    def create_episode_gif(self, n: int):
        for i in range(n):
            episode_letter = chr(97 + i)

            # collect an episode
            done = False
            frames = []
            state = tf.reshape(self.env.reset(), [1, -1])
            while not done:
                frames.append(self.env.render(mode="rgb_array"))

                action, _ = self.agent.act(state)
                observation, reward, done, _ = self.env.step(action.numpy())
                state = tf.reshape(observation, [1, -1])

            # the figure
            plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
            patch = plt.imshow(frames[0])
            plt.axis('off')

            def animate(i):
                patch.set_data(frames[i])

            anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
            anim.save(f"{self.story_directory}/iteration_{self.agent.iteration}_{episode_letter}.gif", writer='imagemagick', fps=30)

            plt.close()

    def make_metadata(self):
        # TODO
        pass

    def make_hp_box(self):
        # TODO
        pass

    def update_reward_graph(self):
        fig, ax = plt.subplots()

        ax.plot(self.agent.gatherer.mean_episode_reward_per_gathering, label="Average Reward")
        ax.plot(self.agent.gatherer.stdev_episode_reward_per_gathering, label="Standard Deviation")

        ax.set_xticks(list(range(self.agent.iteration)))

        ax.set_title("Mean Rewards and their Standard Deviations for Each Training Cycle.")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Reward")

        ax.legend(loc=2)
        fig.savefig(f"{self.story_directory}/reward_plot.svg", format="svg")

        plt.close(fig)

    def update_story(self):
        story = ""

        with open(f"{PATH_TO_STORIES}/template/head.html") as f:
            story += f.read()
            story += "\n\n"

        # main title
        story += f"<h1 align='center'>A Story About {self.agent.gatherer.env.unwrapped.spec.id}</h1>\n\n"

        # reward plot
        reward_plot_path = self.story_directory + "/reward_plot.svg"
        if os.path.isfile(reward_plot_path):
            story += f"<div class='plot-block'>\n" \
                     f"\t<img src=reward_plot.svg />\n" \
                     f"</div>\n\n"

        gif_files = sorted([fp for fp in os.listdir(self.story_directory) if fp[-4:] == ".gif"],
                           key=lambda f: int(re.search("[0-9]+", f).group(0)))

        last_gif_iteration = -1
        for gif_filename in gif_files:
            gif_filepath = f"{gif_filename}"

            gif_iteration = int(re.search('[0-9]+', gif_filepath).group(0))

            if gif_iteration != last_gif_iteration:
                if last_gif_iteration != -1:
                    story += "</div>\n\n"

                story += f"<div class='iteration-block'>\n" \
                         f"\t<h3>Iteration {gif_iteration}</h3>\n" \
                         f"\t<h5>&mu; = {round(self.agent.gatherer.mean_episode_reward_per_gathering[gif_iteration], 2)}, " \
                         f"&sigma; = {round(self.agent.gatherer.stdev_episode_reward_per_gathering[gif_iteration], 2)}</h5>"

            story += f"\t<img src={gif_filepath} />\n" \

            last_gif_iteration = gif_iteration
        story += "</div>\n\n"

        with open(f"{PATH_TO_STORIES}/template/foot.html") as f:
            story += f.read()

        with open(f"{self.story_directory}/story.html", "w") as f:
            f.write(story)


if __name__ == "__main__":
    tf.enable_eager_execution()

    env = gym.make("LunarLander-v2")
    agent = RandomAgent(env)

    teller = StoryTeller(agent, env, frequency=3)

    for i in range(3):
        teller.create_episode_gif(3)
        agent.iteration += 1

    teller.update_reward_graph()
    teller.update_story()
