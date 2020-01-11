"""Probability distributions used in Stochastic Policies."""

import math
from typing import Union, Tuple

import numpy as np
import tensorflow as tf
import abc


class _PolicyDistribution(abc.ABC):
    """Policy abstract base class.

    Each policy implementation provides basic probabilistic methods to calculate probability density and entropy, both
    in standard and in log space. It furthermore gives the characteristic act method that given the probability
    parameters samples the action and provides the actions probability."""

    @property
    def is_continuous(self):
        return False

    @abc.abstractmethod
    def act(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def sample(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def probability(self, **kwargs):
        """Get the probability of given sample with given distribution parameters. For continuous distributions this is
        the probability density function."""
        pass

    @abc.abstractmethod
    def log_probability(self, **kwargs):
        """Get the logarithmic probability of given sample with given distribution parameters. For continuous
        distributions this is the probability density function.

        The logarithmic probability has better numerical stability properties during later calculations."""
        pass

    @abc.abstractmethod
    def entropy(self):
        pass

    @abc.abstractmethod
    def entropy_from_log(self):
        pass

    @tf.function
    def approximate_kl_divergence(self, log_pa: tf.Tensor, log_pb: tf.Tensor):
        """Approximate KL-divergence between distributions a and b where some sample has probability pa in a
        and pb in b, based on log probabilities."""
        return .5 * tf.reduce_mean(tf.square(log_pa - log_pb))


class CategoricalPolicyDistribution(_PolicyDistribution):
    """Policy implementation fro categorical (also discrete) distributions. That is, this policy is to be used in any
    case where the action space is discrete and the agent thus predicts a pmf over the possible actions."""

    def act(self, log_probabilities: Union[tf.Tensor, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Sample an action from a discrete action distribution predicted by the given policy for a given state."""
        action = self.sample(log_probabilities)
        return action, log_probabilities[0][action]

    def sample(self, log_probabilities):
        assert isinstance(log_probabilities, tf.Tensor) or isinstance(log_probabilities, np.ndarray), \
            f"Policy methods (act_discrete) require Tensors or Numpy Arrays as input, " \
            f"not {type(log_probabilities).__name__}."

        if tf.rank(log_probabilities) == 3:
            # there appears to be a sequence dimension
            assert log_probabilities.shape[
                       1] == 1, "Policy actions can only be selected for a single timestep, but the " \
                                "dimensionality of the given tensor is more than 1 at rank 1."

            log_probabilities = tf.squeeze(log_probabilities, axis=1)

        action = tf.random.categorical(log_probabilities, 1)[0][0]
        return action.numpy()

    def probability(self, **kwargs):
        raise NotImplementedError("A categorical distribution has no pdf.")

    def log_probability(self, **kwargs):
        raise NotImplementedError("A categorical distribution has no pdf.")

    @tf.function
    def entropy(self, pmf: tf.Tensor):
        """Calculate entropy of a categorical distribution, where the pmf is given as log probabilities."""
        return - tf.reduce_sum(tf.math.log(pmf) * pmf, axis=-1)

    @tf.function
    def entropy_from_log(self, pmf: tf.Tensor):
        """Calculate entropy of a categorical distribution, where the pmf is given as log probabilities."""
        return - tf.reduce_sum(tf.exp(pmf) * pmf, axis=-1)


class _ContinuousPolicyDistribution(_PolicyDistribution, abc.ABC):

    @property
    def is_continuous(self):
        return True


class GaussianPolicyDistribution(_ContinuousPolicyDistribution):
    """Gaussian Probability Distribution."""

    def act(self, means: tf.Tensor, log_stdevs: tf.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Sample an action from a gaussian action distribution defined by the means and log standard deviations."""
        actions = tf.random.normal(means.shape, means, tf.exp(log_stdevs))
        probabilities = self.log_probability(actions, means=means, log_stdevs=log_stdevs)

        return tf.reshape(actions, [-1]).numpy(), tf.squeeze(probabilities).numpy()

    def sample(self, means: tf.Tensor, log_stdevs: tf.Tensor):
        action = tf.random.normal(means.shape, means, tf.exp(log_stdevs))
        return tf.reshape(action, [-1]).numpy()

    @tf.function
    def probability(self, samples: tf.Tensor, means: tf.Tensor, stdevs: tf.Tensor):
        """Calculate probability density for a given batch of potentially joint Gaussian PDF."""
        samples_transformed = (samples - means) / stdevs
        pdf = (tf.exp(-(tf.pow(samples_transformed, 2) / 2)) / tf.sqrt(2 * math.pi)) / stdevs
        return tf.math.reduce_prod(pdf, axis=-1)

    @tf.function
    def log_probability(self, samples: tf.Tensor, means: tf.Tensor, log_stdevs: tf.Tensor):
        """Calculate log probability density for a given batch of potentially joint Gaussian PDF.

        Input Shapes:
            - all: (B, A) or (B, S, A)

        Output Shapes:
            - all: (B) or (B, S)
        """
        log_likelihoods = (- tf.reduce_sum(log_stdevs, axis=-1)
                           - tf.math.log(2 * np.pi)
                           - (0.5 * tf.reduce_sum(tf.square(((samples - means) / tf.exp(log_stdevs))), axis=-1)))

        # log_likelihoods = tf.math.log(gaussian_pdf(samples, means, tf.exp(log_stdevs)))

        return log_likelihoods

    @tf.function
    def entropy(self, stdevs: tf.Tensor):
        """Calculate the joint entropy of Gaussian random variables described by their log standard deviations.

        Input Shape: (B, A) or (B, S, A) for recurrent

        Since the given r.v.'s are independent, the subadditivity property of entropy narrows down to an equality
        of the joint entropy and the sum of marginal entropies.
        """
        entropy = .5 * tf.math.log(2 * math.pi * math.e * tf.pow(stdevs, 2))
        return tf.reduce_sum(entropy, axis=-1)

    @tf.function
    def approx_entropy_from_log(self, log_stdevs: tf.Tensor):
        """Calculate the joint entropy of Gaussian random variables described by their log standard deviations.

        Input Shape: (B, A) or (B, S, A) for recurrent

        Since the given r.v.'s are independent, the subadditivity property of entropy narrows down to an equality
        of the joint entropy and the sum of marginal entropies.
        """
        entropy = log_stdevs
        return tf.reduce_sum(entropy, axis=-1)

    @tf.function
    def entropy_from_log(self, log_stdevs: tf.Tensor):
        """Calculate the joint entropy of Gaussian random variables described by their log standard deviations.

        Input Shape: (B, A) or (B, S, A) for recurrent

        Since the given r.v.'s are independent, the subadditivity property of entropy narrows down to an equality
        of the joint entropy and the sum of marginal entropies.
        """
        entropy = .5 * (tf.math.log(math.pi * 2) + (tf.multiply(2.0, log_stdevs) + 1.0))
        return tf.reduce_sum(entropy, axis=-1)


class BetaPolicyDistribution(_ContinuousPolicyDistribution):
    """Beta Distribution."""

    def __init__(self):
        pass

    def act(self, *args, **kwargs):
        pass

    def sample(self, *args, **kwargs):
        pass

    def prob(self, action, *args, **kwargs):
        pass

    def probability(self, **kwargs):
        pass

    def log_probability(self, **kwargs):
        pass

    def entropy(self):
        pass

    def entropy_from_log(self):
        pass


_distribution_short_name_map = {
    "gaussian": GaussianPolicyDistribution(),
    "discrete": CategoricalPolicyDistribution(),
    "categorical": CategoricalPolicyDistribution(),
    "beta": BetaPolicyDistribution()
}


def get_distribution_by_short_name(name: str) -> _PolicyDistribution:
    """Get a policy distribution object based on a short name identifier."""
    if name not in _distribution_short_name_map.keys():
        raise ValueError("Unknown distribution type.")

    return _distribution_short_name_map[name]
