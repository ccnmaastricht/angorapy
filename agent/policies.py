"""Probability distributions used in Stochastic Policies."""

import abc
import math
from typing import Union, Tuple

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from agent.layers import StdevLayer
from utilities.const import NUMPY_FLOAT_PRECISION
from utilities.util import env_extract_dims


class BasePolicyDistribution(abc.ABC):
    """Abstract base class for policy distributions.

    Each policy implementation provides basic probabilistic methods to calculate probability density and entropy, both
    in standard and in log space. It furthermore gives the characteristic act method that given the probability
    parameters samples the action and provides the actions probability."""

    def __init__(self, env: gym.Env):
        self.state_dim, self.action_dim = env_extract_dims(env)
        self.action_max_values = env.action_space.high
        self.action_min_values = env.action_space.low
        self.action_mm_diff = self.action_max_values - self.action_min_values

    @property
    def is_continuous(self):
        """Indicate whether the distribution is continuous."""
        return False

    @property
    @abc.abstractmethod
    def has_log_params(self):
        """Indicate whether the parameters of the distribution are expected to be given in log space."""
        pass

    @abc.abstractmethod
    def act(self, *args, **kwargs):
        """Sample an action from the distribution and return it alongside its probability."""
        pass

    @abc.abstractmethod
    def sample(self, *args, **kwargs):
        """Sample an action from the distribution."""
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
        """Calculate the entropy of the distribution based on log parameters."""
        pass

    @tf.function
    def approximate_kl_divergence(self, log_pa: tf.Tensor, log_pb: tf.Tensor):
        """Approximate KL-divergence between distributions a and b where some sample has probability pa in a
        and pb in b, based on log probabilities."""
        return .5 * tf.reduce_mean(tf.square(log_pa - log_pb))

    @abc.abstractmethod
    def build_action_head(self, n_actions: int, input_shape: tuple, batch_size: Union[int, None]):
        """Build the action head of a policy using this distribution."""
        pass


class CategoricalPolicyDistribution(BasePolicyDistribution):
    """Policy implementation fro categorical (also discrete) distributions. That is, this policy is to be used in any
    case where the action space is discrete and the agent thus predicts a pmf over the possible actions."""

    @property
    def has_log_params(self):
        """Categorical distribution expects pmf in log space."""
        return True

    def act(self, log_probabilities: Union[tf.Tensor, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Sample an action from a discrete action distribution predicted by the given policy for a given state."""
        action = self.sample(log_probabilities)
        return action, log_probabilities[0][action]

    def sample(self, log_probabilities):
        """Sample an action from the distribution."""
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
        """Not Implemented"""
        raise NotImplementedError("A categorical distribution has no pdf.")

    def log_probability(self, **kwargs):
        """Not implemented"""
        raise NotImplementedError("A categorical distribution has no pdf.")

    @tf.function
    def _entropy_from_pmf(self, pmf: tf.Tensor):
        """Calculate entropy of a categorical distribution from raw pmf."""
        return - tf.reduce_sum(tf.math.log(pmf) * pmf, axis=-1)

    @tf.function
    def _entropy_from_log_pmf(self, pmf: tf.Tensor):
        """Calculate entropy of a categorical distribution, where the pmf is given as log probabilities."""
        return - tf.reduce_sum(tf.exp(pmf) * pmf, axis=-1)

    @tf.function
    def entropy(self, pmf: tf.Tensor):
        """Calculate entropy of a categorical distribution, where the pmf is given as log probabilities."""
        return - self._entropy_from_log_pmf(pmf)

    def build_action_head(self, n_actions: int, input_shape: tuple, batch_size: int):
        """Build a discrete action head as a log softmax output layer."""
        inputs = tf.keras.Input(batch_shape=(batch_size,) + tuple(input_shape))

        x = tf.keras.layers.Dense(n_actions,
                                  kernel_initializer=tf.keras.initializers.Orthogonal(0.01),
                                  bias_initializer=tf.keras.initializers.Constant(0.0))(inputs)
        x = tf.nn.log_softmax(x, name="log_likelihoods")

        return tf.keras.Model(inputs=inputs, outputs=x, name="discrete_action_head")


class BaseContinuousPolicyDistribution(BasePolicyDistribution, abc.ABC):
    """Abstract base class of continuous policy distributions."""

    @property
    def is_continuous(self):
        """Indicate that the distribution is continuous."""
        return True


class GaussianPolicyDistribution(BaseContinuousPolicyDistribution):
    """Gaussian Probability Distribution."""

    @property
    def has_log_params(self):
        """Gaussian Distribution expects standard deviation in log space"""
        return True

    def act(self, means: tf.Tensor, log_stdevs: tf.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Sample an action from a gaussian action distribution defined by the means and log standard deviations."""
        actions = tf.random.normal(means.shape, means, tf.exp(log_stdevs))
        probabilities = self.log_probability(actions, means=means, log_stdevs=log_stdevs)

        return tf.reshape(actions, [-1]).numpy(), tf.squeeze(probabilities).numpy()

    def sample(self, means: tf.Tensor, log_stdevs: tf.Tensor):
        """Sample from the Gaussian distribution."""
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
    def _entropy_from_params(self, stdevs: tf.Tensor):
        """Calculate the joint entropy of Gaussian random variables described by their standard deviations.

        Input Shape: (B, A) or (B, S, A) for recurrent

        Since the given r.v.'s are independent, the subadditivity property of entropy narrows down to an equality
        of the joint entropy and the sum of marginal entropies.
        """
        entropy = .5 * tf.math.log(2 * math.pi * math.e * tf.pow(stdevs, 2))
        return tf.reduce_sum(entropy, axis=-1)

    @tf.function
    def _entropy_from_log_params(self, log_stdevs: tf.Tensor):
        """Calculate the joint entropy of Gaussian random variables described by their log standard deviations.

        Input Shape: (B, A) or (B, S, A) for recurrent

        Since the given r.v.'s are independent, the subadditivity property of entropy narrows down to an equality
        of the joint entropy and the sum of marginal entropies.
        """
        entropy = .5 * (tf.math.log(math.pi * 2) + (tf.multiply(2.0, log_stdevs) + 1.0))
        return tf.reduce_sum(entropy, axis=-1)

    @tf.function
    def _approx_entropy_from_log(self, log_stdevs: tf.Tensor):
        """Calculate the joint entropy of Gaussian random variables described by their log standard deviations, but in
        an approximation. Essentially this removes any unnecessary scaling calculations and only leaves the bare
        log standard deviations as the entropy of the distribution.

        Input Shape: (B, A) or (B, S, A) for recurrent

        Since the given r.v.'s are independent, the subadditivity property of entropy narrows down to an equality
        of the joint entropy and the sum of marginal entropies.
        """
        return tf.reduce_sum(log_stdevs, axis=-1)

    @tf.function
    def entropy(self, params: tf.Tensor):
        """Calculate the joint entropy of Gaussian random variables described by their log standard deviations.

        Input Shape: (B, A) or (B, S, A) for recurrent

        Since the given r.v.'s are independent, the subadditivity property of entropy narrows down to an equality
        of the joint entropy and the sum of marginal entropies.
        """
        log_stdevs = params[1]
        return self._entropy_from_log_params(log_stdevs)

    def build_action_head(self, n_actions, input_shape, batch_size, stdevs_from_latent=False):
        """Build a policy head for the gaussian distribution, for mean and stdev prediction."""
        inputs = tf.keras.Input(batch_shape=(batch_size,) + tuple(input_shape))

        means = tf.keras.layers.Dense(n_actions, name="means",
                                      kernel_initializer=tf.keras.initializers.Orthogonal(0.01),
                                      bias_initializer=tf.keras.initializers.Constant(0.0))(inputs)
        if stdevs_from_latent:
            stdevs = tf.keras.layers.Dense(n_actions, name="log_stdevs",
                                           kernel_initializer=tf.keras.initializers.Orthogonal(0.01),
                                           bias_initializer=tf.keras.initializers.Constant(0.0))(inputs)
        else:
            stdevs = StdevLayer(n_actions, name="log_stdevs")(means)

        return tf.keras.Model(inputs=inputs, outputs=[means, stdevs], name="gaussian_action_head")


class BetaPolicyDistribution(BaseContinuousPolicyDistribution):
    """Beta Distribution."""

    def __init__(self, env: gym.Env):
        super().__init__(env)

        # assure that rescaling is possible
        assert not np.any(np.isinf(self.action_min_values)) and not np.any(np.isinf(self.action_max_values))

    @property
    def has_log_params(self):
        """Beta Distribution expects all parameters on standard scale."""
        return False

    def act(self, alphas: tf.Tensor, betas: tf.Tensor):
        """Sample an action from a beta distribution."""
        actions = np.random.beta(alphas, betas).astype("float32")
        actions = self._scale_sample_to_action_range(np.reshape(actions, [-1])).numpy()
        probabilities = tf.squeeze(self.log_probability(actions, alphas, betas)).numpy()

        return actions, probabilities

    def sample(self, alphas: tf.Tensor, betas: tf.Tensor):
        """Sample from the Beta distribution."""
        actions = tfp.distributions.Beta(alphas, betas).sample(alphas.shape)
        actions = self._scale_sample_to_action_range(tf.reshape(actions, [-1])).numpy()

        return actions

    @tf.function
    def _scale_sample_to_action_range(self, sample) -> tf.Tensor:
        return tf.add(tf.multiply(sample, self.action_mm_diff), self.action_min_values)

    @tf.function
    def _scale_sample_to_distribution_range(self, sample) -> tf.Tensor:
        # clipping just to, you know, be sure
        return tf.clip_by_value(tf.divide(tf.subtract(sample, self.action_min_values), self.action_mm_diff), 0, 1)

    @tf.function
    def probability(self, samples: tf.Tensor, alphas: tf.Tensor, betas: tf.Tensor):
        """Probability density of the Beta distribution."""
        samples = self._scale_sample_to_distribution_range(samples)

        top = tf.pow(samples, tf.subtract(alphas, 1.)) * tf.pow(tf.subtract(1., samples), tf.subtract(betas, 1.))
        bab = tf.multiply(tf.exp(tf.math.lgamma(alphas)),
                          tf.exp(tf.math.lgamma(betas)) / tf.exp(tf.math.lgamma(tf.add(alphas, betas))))

        return tf.math.reduce_prod(top / bab, axis=-1)

    @tf.function
    def log_probability(self, samples: tf.Tensor, alphas: tf.Tensor, betas: tf.Tensor):
        """Log probability utilizing the fact that tensorflow directly returns log of gamma function.

        Input Shape: (B, A)
        """
        samples = self._scale_sample_to_distribution_range(samples)

        top = tf.pow(samples, tf.subtract(alphas, 1.)) * tf.pow(
            tf.subtract(1., samples), tf.subtract(betas, 1.))
        log_pdf = tf.math.log(top) - tf.math.lgamma(alphas) - tf.math.lgamma(betas) + tf.math.lgamma(alphas + betas)

        return tf.math.reduce_sum(log_pdf, axis=-1)

    @tf.function
    def entropy(self, params: Tuple[tf.Tensor, tf.Tensor]):
        """Entropy of the beta distribution."""
        return self._entropy_from_params(params)

    @tf.function
    def _entropy_from_params(self, params: Tuple[tf.Tensor, tf.Tensor]):
        """Entropy of the beta distribution"""
        alphas, betas = params

        bab = tf.multiply(tf.exp(tf.math.lgamma(alphas)),
                          tf.exp(tf.math.lgamma(betas)) / tf.exp(tf.math.lgamma(tf.add(alphas, betas))))
        a = tf.multiply(tf.subtract(alphas, 1.), tf.math.polygamma(0., alphas))
        b = tf.multiply(tf.subtract(betas, 1.), tf.math.polygamma(0., betas))
        ab = tf.multiply(tf.subtract(alphas + betas, 2.), tf.math.polygamma(0., tf.add(alphas, betas)))

        return tf.reduce_sum(tf.math.log(bab) - a - b + ab, axis=-1)

    def build_action_head(self, n_actions, input_shape, batch_size):
        """Build a policy head for the beta distribution, for alpha and beta prediction.

        """
        inputs = tf.keras.Input(batch_shape=(batch_size,) + tuple(input_shape))

        alphas = tf.keras.layers.Dense(n_actions, name="alphas", activation="softplus",
                                       kernel_initializer=tf.keras.initializers.Orthogonal(1),
                                       bias_initializer=tf.keras.initializers.Constant(0.0))(inputs)

        betas = tf.keras.layers.Dense(n_actions, name="betas", activation="softplus",
                                      kernel_initializer=tf.keras.initializers.Orthogonal(1),
                                      bias_initializer=tf.keras.initializers.Constant(0.0))(inputs)

        # guarantee a, b > 1 so that the distribution is concave and unimodal, see Chou, Maturana & Scherer (2017)
        alphas = tf.add(alphas, 1.)
        betas = tf.add(betas, 1.)

        return tf.keras.Model(inputs=inputs, outputs=[alphas, betas], name="beta_action_head")


_distribution_short_name_map = {
    "gaussian": GaussianPolicyDistribution,
    "discrete": CategoricalPolicyDistribution,
    "categorical": CategoricalPolicyDistribution,
    "beta": BetaPolicyDistribution
}


def get_distribution_by_short_name(name: str) -> type(BasePolicyDistribution):
    """Get a policy distribution object based on a short name identifier."""
    if name not in _distribution_short_name_map.keys():
        raise ValueError("Unknown distribution type.")

    return _distribution_short_name_map[name]
