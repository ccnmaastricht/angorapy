import abc
import sys
from collections import namedtuple
from typing import List, Dict

import numpy as np
import gymnasium as gym

from angorapy.common.senses import Sensation
from angorapy.common.const import NP_FLOAT_PREC, EPSILON
from angorapy.utilities.dtypes import StepTuple
from angorapy.utilities.core import env_extract_dims


PostProcessorSerialization = namedtuple("PostProcessorSerialization", ["class_name", "env_id", "data"])


class BasePostProcessor(abc.ABC):
    """Abstract base class for postprocessors."""

    def __init__(self, env_name: str, state_dim, n_actions):
        self.n = 1e-4  # make this at least epsilon so that first measure is not all zeros
        self.previous_n = self.n  # TODO there has to be a better way to do this

        # extract env info
        self.env_name = env_name
        self.state_dim, self.number_of_actions = state_dim, n_actions

    @abc.abstractmethod
    def __add__(self, other):
        pass

    def __repr__(self):
        return self.__class__.__name__

    @property
    def name(self):
        """The name of the postprocessors."""
        return self.__class__.__name__

    def transform(self, step_result: StepTuple, **kwargs) -> StepTuple:
        """Transform the step results."""
        pass

    def update(self, **kwargs):
        """Just for the sake of interchangeability, all postprocessors have an update method even if they do not update."""
        pass

    def warmup(self, env: "TaskWrapper", n_steps=10):
        """Warm up the postprocessor for n steps."""
        pass

    @abc.abstractmethod
    def serialize(self) -> dict:
        """Serialize the postprocessor to allow for saving its data in a file."""
        pass

    @staticmethod
    def from_serialization(serialization: PostProcessorSerialization):
        """Create postprocessor from a serialization."""
        return getattr(sys.modules[__name__], serialization.class_name).recover(serialization.data)

    @classmethod
    @abc.abstractmethod
    def recover(cls, env_id, state_dim, n_actions, serialization_data: list):
        """Recover from serialization."""
        pass

    @staticmethod
    def from_collection(collection_of_postprocessors: List["BasePostProcessor"]) -> "BasePostProcessor":
        """Merge a list of postprocessors into one new postprocessor of the same type."""
        assert len(set([type(w) for w in collection_of_postprocessors])) == 1, \
            "All postprocessors need to have the same type."

        new_postprocessors = collection_of_postprocessors[0]
        for postprocessor in collection_of_postprocessors[1:]:
            new_postprocessors += postprocessor

        return new_postprocessors

    def correct_sample_size(self, deduction):
        """Deduce the given number from the sample counter."""
        self.n = self.n - deduction


class BaseRunningMeanPostProcessor(BasePostProcessor, abc.ABC):
    """Abstract base class for postprocessors implementing a running mean over some statistic."""

    mean: Dict[str, np.ndarray]
    variance: Dict[str, np.ndarray]

    def __add__(self, other) -> "BaseRunningMeanPostProcessor":
        pop = self.__class__(self.env_name, self.state_dim, self.number_of_actions)
        pop.n = self.n + other.n

        for name in pop.mean.keys():
            pop.mean[name] = (self.n / pop.n) * self.mean[name] + (other.n / pop.n) * other.mean[name]
            pop.variance[name] = (self.n * (self.variance[name] + (self.mean[name] - pop.mean[name]) ** 2)
                                 + other.n * (other.variance[name] + (other.mean[name] - pop.mean[name]) ** 2)) / pop.n

        return pop

    def update(self, observation: dict) -> None:
        """Update the mean(serialization) and variance(serialization) of the tracked statistic based on the new sample.

        Simplification of https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm.."""
        self.n += 1

        for name, obs in filter(lambda o: isinstance(o[1], (int, float)) or len(o[1].shape) in [0, 1],
                                observation.items()):
            delta = obs - self.mean[name]
            m_a = self.variance[name] * (self.n - 1)

            self.mean[name] = np.array(self.mean[name] + delta * (1 / self.n), dtype=NP_FLOAT_PREC)
            self.variance[name] = np.array((m_a + np.square(delta) * (self.n - 1) / self.n) / self.n,
                                           dtype=NP_FLOAT_PREC)

    def serialize(self) -> PostProcessorSerialization:
        """Serialize the postprocessor to allow for saving it in a file."""
        return PostProcessorSerialization(self.__class__.__name__,
                                          self.env_name,
                                          [self.n,
                                           self.previous_n,
                                         {n: m.tolist() for n, m in self.mean.items()},
                                         {n: v.tolist() for n, v in self.variance.items()}])

    @classmethod
    def recover(cls, env_id, state_dim, n_actions, data: PostProcessorSerialization):
        """Recover a running mean postprocessors from its serialization"""
        postprocessor = cls(env_id, state_dim, n_actions)

        if len(data) == 3:  # backwards compatibility
            n, means, variances = data
        else:
            n, previous_n, means, variances = data
            postprocessor.previous_n = previous_n

        postprocessor.n = n

        # backwards compatibility
        compatible_data_means = {name if name != "somatosensation" else "touch": l for name, l in means.items()}
        compatible_data_variances = {name if name != "somatosensation" else "touch": l for name, l in variances.items()}
        compatible_data_means = {name if name != "asynchronous" else "asymmetric": l for name, l in compatible_data_means.items()}
        compatible_data_variances = {name if name != "asynchronous" else "asymmetric": l for name, l in compatible_data_variances.items()}

        postprocessor.mean = {name: np.array(l) for name, l in compatible_data_means.items()}
        postprocessor.variance = {name: np.array(l) for name, l in compatible_data_variances.items()}

        return postprocessor

    def simplified_mean(self) -> Dict[str, float]:
        """Get a simplified, one dimensional mean by meaning any means."""
        return {n: np.mean(m).item() for n, m in self.mean.items()}

    def simplified_variance(self) -> Dict[str, float]:
        """Get a simplified, one dimensional variance by meaning any variances."""
        return {n: np.mean(v).item() for n, v in self.variance.items()}

    def simplified_stdev(self) -> Dict[str, float]:
        """Get a simplified, one dimensional stdev by meaning any variances and taking their square root."""
        return {n: np.sqrt(np.mean(v)).item() for n, v in self.variance.items()}


class StateNormalizer(BaseRunningMeanPostProcessor):
    """Postprocessor for state normalization using running mean and variance estimations."""

    def __init__(self, env_name: str, state_dim, n_actions):
        # parse input types into normed shape format
        super().__init__(env_name, state_dim, n_actions)

        self.shapes = self.state_dim

        self.mean = {k: np.zeros(i_shape, NP_FLOAT_PREC) for k, i_shape in self.shapes.items() if len(i_shape) == 1}
        self.variance = {k: np.ones(i_shape, NP_FLOAT_PREC) for k, i_shape in self.shapes.items() if len(i_shape) == 1}

        assert len(self.mean) > 0 and len(self.variance) > 0, "Initialized StateNormalizer got no vector states."

    def transform(self, step_result: StepTuple, update=True) -> StepTuple:
        """Normalize a given batch of 1D tensors and update running mean and std."""
        o, r, terminated, truncated, info = step_result

        if update:
            self.update(o.dict())

        normed_o = {}
        # for sense_name, sense_value in filter(lambda a: len(a[1].shape) == 1, o.dict().items()):
        for sense_name, sense_value in o.dict().items():
            if len(sense_value.shape) == 1:
                normed_o[sense_name] = np.clip(
                    (sense_value - self.mean[sense_name]) / (np.sqrt(self.variance[sense_name] + EPSILON)), -10., 10.)
            else:
                # assuming visual RGB values
                normed_o[sense_name] = np.divide(sense_value, 255)

        normed_o = Sensation(**normed_o)

        return normed_o, r, terminated, truncated, info

    def warmup(self, env: "TaskWrapper", n_steps=10):
        """Warmup the postprocessor by sampling the observation space."""
        return  # todo we cannot do this like this, the update will be based on a transformed input this way
        env.reset()
        for i in range(n_steps):
            self.update(env.step(env.action_space.sample())[0].dict())


class RewardNormalizer(BaseRunningMeanPostProcessor):
    """Postprocessor for reward normalization using running mean and variance estimations."""

    def __init__(self, env_name: str, state_dim, n_action):
        super().__init__(env_name, state_dim, n_action)

        self.mean = {"reward": np.array(0, NP_FLOAT_PREC)}
        self.variance = {"reward": np.array(1, NP_FLOAT_PREC)}
        self.ret = np.float64(0)

    def transform(self, step_tuple: StepTuple, update=True) -> StepTuple:
        """Normalize a given batch of 1D tensors and update running mean and std."""
        o, r, terminated, truncated, info = step_tuple

        if r is None:
            return o, r, terminated, truncated, info  # skip

        # update based on cumulative discounted reward
        if update:
            self.ret = 0.99 * self.ret + r
            self.update({"reward": self.ret})

        # normalize
        r = np.clip(r / (np.sqrt(self.variance["reward"] + EPSILON)), -10., 10.)

        if terminated or truncated:
            self.ret = 0.

        return o, r, terminated, truncated, info

    def warmup(self, env: "TaskWrapper", n_steps=10):
        """Warmup the postprocessor by randomly stepping the environment through action space sampling."""
        env.reset()
        for i in range(n_steps):
            self.update({"reward": env.step(env.action_space.sample())[1]})


def merge_postprocessors(postprocessors: List[BasePostProcessor]) -> BasePostProcessor:
    """Merge a list of postprocessors into a single postprocessor.

    Args:
        postprocessors:           list of postprocessors to merge
    """
    assert all(type(t) is type(postprocessors[0]) for t in postprocessors), \
        "To merge postprocessors, they must be of same type."

    # merge the list of postprocessors into a single postprocessor
    merged_postprocessor = BasePostProcessor.from_collection(postprocessors)

    # adjust for overcounting
    merged_postprocessor.correct_sample_size((len(postprocessors) - 1) * postprocessors[0].previous_n)

    # record the new n for next sync
    merged_postprocessor.previous_n = merged_postprocessor.n

    return merged_postprocessor


def postprocessors_from_serializations(list_of_serializations: List[PostProcessorSerialization]) -> List[BasePostProcessor]:
    """From a list of PostProcessorSerializations recover the respective postprocessors."""
    postprocessors = []
    reference_env = gym.make(PostProcessorSerialization(*list_of_serializations[0]).env_id, render_mode="rgb_array")  # todo could leak memory
    state_dim, n_actions = env_extract_dims(reference_env)

    for cereal in list_of_serializations:
        cereal = PostProcessorSerialization(*cereal)

        class_name = cereal.class_name

        # backward compatibility with formerly used class names
        if class_name == "StateNormalizationTransformer":
            class_name = "StateNormalizer"
        elif class_name == "RewardNormalizationTransformer":
            class_name = "RewardNormalizer"

        postprocessors.append(
            getattr(sys.modules[__name__], class_name).recover(cereal.env_id, state_dim, n_actions, cereal.data)
        )

    return postprocessors
