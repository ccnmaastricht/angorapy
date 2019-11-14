#!/usr/bin/env python
"""Custom datatypes form simplifying data communication."""
from collections import namedtuple

ExperienceBuffer = namedtuple("ExperienceBuffer", ["states", "actions", "action_probabilities", "returns", "advantages",
                                                   "episodes_completed", "episode_rewards", "episode_lengths"])
StatBundle = namedtuple("StatBundle", ["numb_completed_episodes", "numb_processed_frames",
                                       "episode_rewards", "episode_lengths"])
ModelTuple = namedtuple("ModelTuple", ["model_builder", "weights"])