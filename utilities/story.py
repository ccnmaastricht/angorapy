#!/usr/bin/env python
"""Methods for creating a story about a training process."""
import os
import re
import time

import gym
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
from gym.spaces import Box
from matplotlib import animation

from agent.policy import act_discrete, act_continuous
from agent.ppo import PPOAgent

PATH_TO_STORIES = "stories/"
TEMPLATE_PATH = PATH_TO_STORIES + "/template/"


def scale(vector):
    return (numpy.array(vector) - min(vector)) / (max(vector) - min(vector))


class StoryTeller:

    def __init__(self, agent: PPOAgent, env: gym.Env, frequency: int, id=None):
        self.agent = agent
        self.env = env

        self.frequency = frequency
        self.continuous_control = isinstance(self.env.action_space, Box)

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
        act = act_continuous if self.continuous_control else act_discrete

        for i in range(n):
            episode_letter = chr(97 + i)

            # collect an episode
            done = False
            frames = []
            state = tf.expand_dims(self.env.reset().astype(numpy.float32), axis=0)
            while not done:
                frames.append(self.env.render(mode="rgb_array"))

                action, _ = act(self.agent.policy, state)
                observation, reward, done, _ = self.env.step(action)
                state = tf.expand_dims(observation.astype(numpy.float32), axis=0)

            # the figure
            plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
            patch = plt.imshow(frames[0], cmap="Greys" if len(frames[0].shape) == 2 else None)
            plt.axis('off')

            def animate(i):
                patch.set_data(frames[i])

            anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
            anim.save(f"{self.story_directory}/iteration_{self.agent.iteration}_{episode_letter}.gif",
                      writer='imagemagick', fps=30)

            plt.close()

    def make_metadata(self):
        # TODO
        pass

    def make_hp_box(self):
        with open(TEMPLATE_PATH + "hp_box.html") as f:
            tpl = f.read()

        relevant_hps = [
            ("CONTINUOUS", self.agent.continuous_control),
            ("LEARNING RATE", self.agent.learning_rate_pi),
            ("EPSILON CLIP", self.agent.clip),
            ("ENTROPY COEFFICIENT", self.agent.c_entropy),
        ]

        hplist = ""
        for p, v in relevant_hps:
            hplist += f"<div class='hp-element'>{p}: {v}</div>\n"

        box = re.sub("%HPS%", hplist, tpl)

        return box

    def update_graphs(self):
        # reward plot
        fig, ax = plt.subplots()
        ax.set_title("Mean Rewards and Episode Lengths for Each Training Cycle.")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Accumulative Reward")
        twin_ax = ax.twinx()
        twin_ax.set_ylabel("Episode Steps")

        r_line = ax.plot(self.agent.cycle_reward_history, label="Average Reward", color="orange")
        l_line = twin_ax.plot(self.agent.cycle_length_history, label="Episode Length", color="blue")
        lines = r_line + l_line
        labels = [l.get_label() for l in lines]

        ax.legend(lines, labels, loc=2)
        fig.savefig(f"{self.story_directory}/reward_plot.svg", format="svg")

        plt.close(fig)

        # entropy, loss plot
        fig, ax = plt.subplots()
        ax.set_title("Entropy and Loss.")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        twin_ax = ax.twinx()
        twin_ax.set_ylabel("Entropy")

        actor_loss_line = ax.plot(scale(self.agent.actor_loss_history), label="Policy Loss (Normalized)")
        critic_loss_line = ax.plot(scale(self.agent.critic_loss_history), label="Critic Loss (Normalized)")
        entropy_line = twin_ax.plot(self.agent.entropy_history, label="Entropy", color="green")
        lines = actor_loss_line + critic_loss_line + entropy_line
        labels = [l.get_label() for l in lines]

        ax.legend(lines, labels, loc=2)
        fig.savefig(f"{self.story_directory}/loss_plot.svg", format="svg")

        plt.close(fig)

    def update_story(self):
        story = ""

        with open(f"{TEMPLATE_PATH}/head.html") as f:
            story += f.read()
            story += "\n\n"

        # main title
        story += f"<h1 align='center'>A Story About {self.agent.env.unwrapped.spec.id}</h1>\n\n"

        # hyperparameters
        story += self.make_hp_box()

        # reward plot
        story += "<div class='plot-wrapper'>"
        reward_plot_path = self.story_directory + "/reward_plot.svg"
        if os.path.isfile(reward_plot_path):
            story += f"<div class='plot-block'>\n" \
                     f"\t<img src=reward_plot.svg />\n" \
                     f"</div>\n\n"
        loss_plot_path = self.story_directory + "/reward_plot.svg"
        if os.path.isfile(loss_plot_path):
            story += f"<div class='plot-block'>\n" \
                     f"\t<img src=loss_plot.svg />\n" \
                     f"</div>\n\n"
        story += "</div>"

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
                         f"\t<h5>&mu; = {0 if self.agent.cycle_reward_history[-1] is None else round(self.agent.cycle_reward_history[gif_iteration], 2)}, " \
                         f"&sigma; = {0 if self.agent.cycle_length_history[-1] is None else round(self.agent.cycle_length_history[gif_iteration], 2)}</h5>"

            story += f"\t<img src={gif_filepath} width=320 height=320 />\n"

            last_gif_iteration = gif_iteration
        story += "</div>\n\n"

        with open(f"{PATH_TO_STORIES}/template/foot.html") as f:
            story += f.read()

        with open(f"{self.story_directory}/story.html", "w") as f:
            f.write(story)
