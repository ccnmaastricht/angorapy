#!/usr/bin/env python
"""Methods for creating a story about a training process."""
import code
import datetime
import os
import socket
import time
from importlib.metadata import version
from inspect import getfullargspec as fargs

import matplotlib
import matplotlib.pyplot as plt
import numpy
import simplejson as json
import tensorflow as tf
from gym.spaces import Box
from matplotlib import animation
from mpi4py import MPI

from angorapy.agent.ppo_agent import PPOAgent
from angorapy.common import const
from angorapy.common.const import PATH_TO_EXPERIMENTS
from angorapy.common.transformers import RewardNormalizationTransformer, StateNormalizationTransformer
from angorapy.common.wrappers import BaseWrapper
from angorapy.models.mighty_maker import get_model_type
from angorapy.utilities.util import add_state_dims, flatten


def scale(vector):
    """Min Max scale a vector."""
    divisor = max(vector) - min(vector)
    return (numpy.array(vector) - min(vector)) / (divisor if divisor != 0 else const.EPSILON)


class Monitor:
    """Monitor for learning progress. Tracks and writes statistics to be parsed by the Flask app."""

    def __init__(self,
                 agent: PPOAgent,
                 env: BaseWrapper = None,
                 frequency: int = 1,
                 gif_every: int = 0,
                 id=None,
                 iterations=None,
                 config_name: str = "unknown",
                 experiment_group: str = "default"):
        self.agent = agent
        self.env = env if env is not None else agent.env
        self.iterations = iterations

        self.experiment_group = experiment_group
        self.config_name = config_name

        self.frequency = frequency
        self.gif_every = gif_every
        self.continuous_control = isinstance(self.env.action_space, Box)

        if id is None:
            self.story_id = agent.agent_id
        else:
            self.story_id = id

        self.story_directory = f"{PATH_TO_EXPERIMENTS}/{self.story_id}/"
        os.makedirs(self.story_directory, exist_ok=True)

        try:
            tf.keras.utils.plot_model(self.agent.joint, to_file=f"{self.story_directory}/model.png", expand_nested=True,
                                      show_shapes=True, dpi=300)
        except:
            print("Could not create model plot.")

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
            state = self.env.reset()
            while not done:
                frames.append(self.env.render(mode="rgb_array"))

                probabilities = flatten(pi.predict(add_state_dims(state, dims=2 if self.agent.is_recurrent else 1)))
                action, _ = self.agent.distribution.act(*probabilities)
                observation, reward, done, _ = self.env.step(
                    numpy.atleast_1d(action) if self.continuous_control else action)
                state = observation

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

    def make_metadata(self, additional_hps: dict = None):
        """Write meta data information about experiment into json file."""

        if additional_hps is None:
            additional_hps = {}

        metadata = dict(
            date=str(datetime.datetime.now()).split(".")[0],
            config=self.config_name,
            host=socket.gethostname(),
            n_cpus=MPI.COMM_WORLD.size,
            n_gpus=self.agent.n_optimizers,
            angorapy_version=version("angorapy"),
            experiment_group=self.experiment_group,
            iterations=self.iterations,
            environment=dict(
                name=self.agent.env_name,
                action_space=str(self.agent.env.action_space),
                observation_space=str(self.agent.env.observation_space),
                deterministic=str(self.agent.env.spec.nondeterministic),
                max_steps=str(self.agent.env.spec.max_episode_steps),
                reward_threshold=str(self.agent.env.spec.reward_threshold),
                max_action_values=str(self.agent.env.action_space.high) if hasattr(self.agent.env.action_space,
                                                                                   "high") else "None",
                min_action_values=str(self.agent.env.action_space.low) if hasattr(self.agent.env.action_space,
                                                                                  "low") else "None",
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
                reward_norming=str(any([isinstance(t, RewardNormalizationTransformer) for t in self.env.transformers])),
                state_norming=str(any([isinstance(t, StateNormalizationTransformer) for t in self.env.transformers])),
                TBPTT_sequence_length=str(self.agent.tbptt_length),
                architecture=self.agent.builder_function_name,
                gatherer=str(self.agent.gatherer_class),
                **additional_hps
            ),
            reward_function=dict(self.agent.env.reward_config if hasattr(self.agent.env, "reward_config") else dict(),
                                 identifier=self.agent.reward_configuration)
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
            preprocessors=self.agent.wrapper_stat_history
        )

        with open(f"{self.story_directory}/progress.json", "w") as f:
            json.dump(progress, f, ignore_nan=True)

    def write_statistics(self):
        """Write general statistics into json file."""
        stat_dict = dict(
            training=dict(
                avg_seconds_per_cycle=str(numpy.mean(self.agent.cycle_timings)) if self.agent.cycle_timings else "N/A",
                total_train_time=str(numpy.sum(self.agent.cycle_timings)) if self.agent.cycle_timings else "0",
            ),
            used_memory=self.agent.used_memory,
            used_gpu_memory=self.agent.used_gpu_memory,
            cycle_timings=self.agent.cycle_timings,
            optimization_timings=self.agent.optimization_timings,
            gathering_timings=self.agent.gathering_timings,
            loaded_at=self.agent.loading_history,
            per_receptor_mean=self.agent.current_per_receptor_mean,
            auxiliary_performances=self.agent.auxiliary_performances
        )

        with open(f"{self.story_directory}/statistics.json", "w") as f:
            json.dump(stat_dict, f, ignore_nan=True)

    def update(self):
        """Update different components of the Monitor."""
        self.write_progress()
        self.write_statistics()
