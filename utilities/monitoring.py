#!/usr/bin/env python
"""Methods for creating a story about a training process."""
import datetime
import json
import os
import time

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
from gym.spaces import Box
from matplotlib import animation

from agent.policy import act_discrete, act_continuous
from agent.ppo import PPOAgent

PATH_TO_EXPERIMENTS = "monitor/experiments/"
matplotlib.use('Agg')


def scale(vector):
    """Min Max scale a vector."""
    return (numpy.array(vector) - min(vector)) / (max(vector) - min(vector))


class Monitor:
    """Monitor for learning progress."""

    def __init__(self, agent: PPOAgent, env: gym.Env, frequency: int, id=None):
        self.agent = agent
        self.env = env

        self.frequency = frequency
        self.continuous_control = isinstance(self.env.action_space, Box)

        if id is None:
            self.story_id = round(time.time())
            self.story_directory = f"{PATH_TO_EXPERIMENTS}/{self.story_id}/"
            os.makedirs(self.story_directory)
        else:
            self.story_id = id
            self.story_directory = f"{PATH_TO_EXPERIMENTS}/{self.story_id}/"
            if not os.path.isdir(self.story_directory):
                raise ValueError("Given ID not found in experiments.")

        tf.keras.utils.plot_model(self.agent.joint, to_file=f"{self.story_directory}/model.png", dpi=300)
        self.make_metadata()
        self.write_progress()

    def create_episode_gif(self, n: int):
        """Make n GIFs with the current policy."""
        act = act_continuous if self.continuous_control else act_discrete

        for i in range(n):
            episode_letter = chr(97 + i)

            # collect an episode
            done = False
            frames = []
            state = tf.expand_dims(self.env.reset().astype(numpy.float32), axis=0)
            while not done:
                frames.append(self.env.render(mode="rgb_array"))

                probabilities = self.agent.policy(state)
                action, _ = act(probabilities)
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
        """Write meta data information about experiment into json file."""
        metadata = dict(
            date=str(datetime.datetime.now()),
            environment=dict(
                name=self.agent.env_name,
                action_space=str(self.agent.env.action_space),
                observation_space=str(self.agent.env.observation_space),
                deterministic=str(self.agent.env.spec.nondeterministic),
                max_steps=str(self.agent.env.spec.max_episode_steps),
                reward_threshold=str(self.agent.env.spec.reward_threshold),
            ),
            hyperparameters=dict(
                continuous=str(self.agent.continuous_control),
                learning_rate=str(self.agent.learning_rate),
                epsilon_clip=str(self.agent.clip),
                entropy_coefficient=str(self.agent.c_entropy.numpy().item()),
                value_coefficient=str(self.agent.c_value.numpy().item()),
                horizon=str(self.agent.horizon),
                workers=str(self.agent.workers),
                discount=str(self.agent.discount),
                GAE_lambda=str(self.agent.lam),
                gradient_clipping=str(self.agent.gradient_clipping),
                clip_values=str(self.agent.clip_values),
                TBPTT_sequence_length=str(self.agent.tbptt_length),
            )
        )

        with open(f"{self.story_directory}/meta.json", "w") as f:
            json.dump(metadata, f)

    def write_progress(self):
        """Write training statistics into json file."""
        progress = dict(
            rewards=self.agent.cycle_reward_history,
            lengths=self.agent.cycle_length_history,
            entropies=self.agent.entropy_history
        )

        with open(f"{self.story_directory}/progress.json", "w") as f:
            json.dump(progress, f)

    def _make_graph(self, ax, lines, labels, name):
        ax.set_title(ax.get_title(),
                     fontdict={'fontsize': "large",
                               'fontweight': "bold",
                               'verticalalignment': 'baseline',
                               'horizontalalignment': "center"}
                     )
        ax.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  fancybox=True, shadow=True, ncol=5)
        plt.savefig(f"{self.story_directory}/{name}.svg", format="svg", bbox_inches="tight")

    def update_graphs(self):
        """Update graphs."""

        # reward plot
        fig, ax = plt.subplots()
        ax.set_title("Mean Rewards and Episode Lengths for Each Training Cycle.")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Accumulative Reward")
        twin_ax = ax.twinx()
        twin_ax.set_ylabel("Episode Steps")

        l_line = twin_ax.plot(self.agent.cycle_length_history, "--", label="Episode Length", color="orange")
        r_line = ax.plot(self.agent.cycle_reward_history, label="Average Reward", color="red")
        lines = r_line + l_line
        labels = [l.get_label() for l in lines]

        self._make_graph(ax, lines, labels, "reward_plot")
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

        self._make_graph(ax, lines, labels, "loss_plot")
        plt.close(fig)

    def update(self):
        """Update different components of the Monitor."""
        self.write_progress()
        self.update_graphs()
