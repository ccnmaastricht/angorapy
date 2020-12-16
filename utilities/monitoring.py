#!/usr/bin/env python
"""Methods for creating a story about a training process."""
import datetime
import socket
import statistics

import simplejson as json
import os
import time
from inspect import getfullargspec as fargs

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
from gym.spaces import Box
from matplotlib import animation

from agent.ppo import PPOAgent
from models import get_model_type
from utilities import const
from utilities.const import PATH_TO_EXPERIMENTS
from utilities.util import parse_state, add_state_dims, flatten
from utilities.wrappers import RewardNormalizationWrapper, StateNormalizationWrapper

matplotlib.use('Agg')


def scale(vector):
    """Min Max scale a vector."""
    divisor = max(vector) - min(vector)
    return (numpy.array(vector) - min(vector)) / (divisor if divisor != 0 else const.EPSILON)


class Monitor:
    """Monitor for learning progress. Tracks and writes statistics to be parsed by the Flask app."""

    def __init__(self, agent: PPOAgent, env: gym.Env, frequency: int, gif_every: int, id=None, iterations=None,
                 config_name: str = "unknown"):
        self.agent = agent
        self.env = env
        self.iterations = iterations

        self.config_name = config_name

        self.frequency = frequency
        self.gif_every = gif_every
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

        # tf.keras.utils.plot_model(self.agent.joint, to_file=f"{self.story_directory}/model.png", expand_nested=True,
        #                           show_shapes=True, dpi=300)

        self.make_metadata()
        self.write_progress()

    def create_episode_gif(self, n: int):
        """Make n GIFs with the current policy."""

        # rebuild model with batch size of 1
        pi, _, _ = self.agent.model_builder(self.env,
                                            **({"bs": 1} if "bs" in fargs(self.agent.model_builder).args else {}))
        pi.set_weights(self.agent.policy.get_weights())

        for j in range(n):
            episode_letter = chr(97 + j)

            # collect an episode
            done = False
            frames = []
            state = parse_state(self.env.reset())
            while not done:
                frames.append(self.env.render(mode="rgb_array"))

                probabilities = flatten(pi.predict(add_state_dims(state, dims=2 if self.agent.is_recurrent else 1)))
                action, _ = self.agent.distribution.act(*probabilities)
                observation, reward, done, _ = self.env.step(
                    numpy.atleast_1d(action) if self.continuous_control else action)
                state = parse_state(observation)

            # the figure
            plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
            patch = plt.imshow(frames[0], cmap="Greys" if len(frames[0].shape) == 2 else None)
            plt.axis('off')

            def _animate(i):
                patch.set_data(frames[i])

            anim = animation.FuncAnimation(plt.gcf(), _animate, frames=len(frames), interval=50)
            anim.save(f"{self.story_directory}/iteration_{self.agent.iteration}_{episode_letter}.gif",
                      writer='pillow',
                      fps=25)


            plt.close()

    def make_metadata(self):
        """Write meta data information about experiment into json file."""
        metadata = dict(
            date=str(datetime.datetime.now()),
            config=self.config_name,
            host=socket.gethostname(),
            iterations=self.iterations,
            environment=dict(
                name=self.agent.env_name,
                action_space=str(self.agent.env.action_space),
                observation_space=str(self.agent.env.observation_space),
                deterministic=str(self.agent.env.spec.nondeterministic),
                max_steps=str(self.agent.env.spec.max_episode_steps),
                reward_threshold=str(self.agent.env.spec.reward_threshold),
                max_action_values=str(self.agent.env.action_space.high) if hasattr(self.agent.env.action_space, "high") else "None",
                min_action_values=str(self.agent.env.action_space.low) if hasattr(self.agent.env.action_space, "low") else "None",
            ),
            hyperparameters=dict(
                continuous=str(self.agent.continuous_control),
                distribution=self.agent.distribution.short_name,
                model=get_model_type(self.agent.model_builder),
                learning_rate=str(self.agent.learning_rate),
                epsilon_clip=str(self.agent.clip),
                entropy_coefficient=str(self.agent.c_entropy.numpy().item()),
                value_coefficient=str(self.agent.c_value.numpy().item()),
                horizon=str(self.agent.horizon),
                workers=str(self.agent.n_workers),
                discount=str(self.agent.discount),
                GAE_lambda=str(self.agent.lam),
                gradient_clipping=str(self.agent.gradient_clipping),
                clip_values=str(self.agent.clip_values),
                reward_norming=str(RewardNormalizationWrapper in self.agent.preprocessor),
                state_norming=str(StateNormalizationWrapper in self.agent.preprocessor),
                TBPTT_sequence_length=str(self.agent.tbptt_length),
                architecture=self.agent.builder_function_name.split("_")[1]
            ),
        )

        with open(f"{self.story_directory}/meta.json", "w") as f:
            json.dump(metadata, f, ignore_nan=True)

    def write_progress(self):
        """Write progress statistics into json file."""
        progress = dict(
            rewards=dict(
                mean=[round(v, 2) if v is not None else v for v in self.agent.cycle_reward_history],
                stdev=[round(v, 2) if v is not None else v for v in self.agent.cycle_reward_std_history],
                last_cycle=self.agent.episode_reward_history[-1] if self.agent.iteration > 1 else []),
            lengths=dict(
                mean=[round(v, 2) if v is not None else v for v in self.agent.cycle_length_history],
                stdev=[round(v, 2) if v is not None else v for v in self.agent.cycle_length_std_history],
                last_cycle=self.agent.episode_length_history[-1] if self.agent.iteration > 1 else []),
            entropies=[round(v, 4) if v is not None else v for v in self.agent.entropy_history],
            vloss=[round(v, 4) if v is not None else v for v in self.agent.value_loss_history],
            ploss=[round(v, 4) if v is not None else v for v in self.agent.policy_loss_history],
            preprocessors=self.agent.preprocessor_stat_history
        )

        with open(f"{self.story_directory}/progress.json", "w") as f:
            json.dump(progress, f, ignore_nan=True)

    def write_statistics(self):
        """Write general statistics into json file."""
        stat_dict = dict(
            training=dict(
                avg_seconds_per_cycle=str(numpy.mean(self.agent.cycle_timings)) if self.agent.cycle_timings else "N/A",
                total_train_time=str(numpy.sum(self.agent.cycle_timings)) if self.agent.cycle_timings else "0",
            )
        )

        with open(f"{self.story_directory}/statistics.json", "w") as f:
            json.dump(stat_dict, f, ignore_nan=True)

    def update(self):
        """Update different components of the Monitor."""
        self.write_progress()
        self.write_statistics()
