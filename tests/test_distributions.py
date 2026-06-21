"""Unit tests for :mod:`angorapy.common.policies` — policy distribution math.

Each policy distribution (Gaussian and Beta for continuous control, Categorical
for discrete) must compute its probability density / mass and its entropy
correctly, in both linear and log space. These quantities feed directly into the
PPO objective (log-probabilities in the ratio, entropy in the exploration bonus),
so the tests validate each against an independent SciPy reference
(``scipy.stats``) rather than against the implementation itself.

Two conventions are checked throughout:

* **log vs. linear consistency** — ``exp(log_probability(.)) == probability(.)``
  and entropy computed from log-parameters matches entropy from raw parameters;
* **per-dimension joint factorization** — for multi-dimensional action spaces the
  joint density is the product (sum in log space) over independent dimensions.

The module also covers :func:`extract_discrete_action_probabilities`, the gather
that picks each taken action's probability out of a batch (and batch+sequence)
of predicted distributions.
"""
import os

from angorapy.agent.utils import extract_discrete_action_probabilities

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gymnasium as gym
import numpy as np
import tensorflow as tf
from gymnasium import spaces
from scipy.stats import norm, entropy, beta

from angorapy.common.policies import GaussianPolicyDistribution, CategoricalPolicyDistribution, \
    BetaPolicyDistribution, MultiCategoricalPolicyDistribution
import pytest


class _MultiDiscreteEnv:
    """Minimal env stub exposing a MultiDiscrete action space (for distribution construction)."""
    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = spaces.MultiDiscrete([3, 3])


# GAUSSIAN
def test_gaussian_pdf():
    """Gaussian density matches SciPy and is log/linear consistent.

    Property
        ``probability`` equals the per-dimension product of ``scipy.stats.norm.pdf``,
        and ``exp(log_probability)`` equals ``probability``.

    Rationale
        The Gaussian log-density is the ratio term for continuous-control PPO; an
        error (e.g. a missing normalization constant or wrong variance handling)
        would silently bias every continuous policy update.
    """
    distro = GaussianPolicyDistribution(gym.make("LunarLanderContinuous-v2"))

    x = tf.convert_to_tensor([[2, 3], [4, 3], [2, 1]], dtype=tf.float32)
    mu = tf.convert_to_tensor([[2, 1], [1, 3], [2, 2]], dtype=tf.float32)
    sig = tf.convert_to_tensor([[2, 2], [1, 2], [2, 1]], dtype=tf.float32)

    result_reference = np.prod(norm.pdf(x, loc=mu, scale=sig), axis=-1)
    result_pdf = distro.probability(x, mu, sig).numpy()
    result_log_pdf = np.exp(distro.log_probability(x, mu, np.log(sig)).numpy())

    assert np.allclose(result_reference, result_pdf), "Gaussian PDF returns wrong Result"
    assert np.allclose(result_pdf, result_log_pdf), "Gaussian Log PDF returns wrong Result"


def test_gaussian_entropy():
    """Gaussian entropy matches SciPy from both raw and log parameters.

    Property
        The summed per-dimension ``scipy.stats.norm.entropy`` equals both the
        entropy computed from raw ``sigma`` and the entropy computed from
        ``log(sigma)``.

    Rationale
        Entropy is the exploration-bonus term in PPO; a wrong constant or a
        log/linear mismatch would mis-scale exploration without crashing.
    """
    distro = GaussianPolicyDistribution(gym.make("LunarLanderContinuous-v2"))

    mu = tf.convert_to_tensor([[2.0, 3.0], [2.0, 1.0]], dtype=tf.float32)
    sig = tf.convert_to_tensor([[1.0, 1.0], [1.0, 5.0]], dtype=tf.float32)

    result_reference = np.sum(norm.entropy(loc=mu, scale=sig), axis=-1)
    result_log = distro.entropy([mu, np.log(sig)]).numpy()
    result = distro._entropy_from_params(sig).numpy()

    assert np.allclose(result_reference, result), "Gaussian entropy returns wrong result"
    assert np.allclose(result_log, result_reference), "Gaussian entropy from log returns wrong result"


# BETA
def test_beta_pdf():
    """Beta density matches SciPy (after range scaling) and is log/linear consistent.

    Property
        ``probability`` and ``exp(log_probability)`` both equal the per-dimension
        product of ``scipy.stats.beta.pdf`` evaluated on the sample after it is
        rescaled into the Beta support via ``_scale_sample_to_distribution_range``.

    Rationale
        The Beta policy bounds actions to a finite interval; the sample-rescaling
        is an easy step to get wrong, and an error there corrupts both the density
        and the resulting gradients.
    """
    distro = BetaPolicyDistribution(gym.make("LunarLanderContinuous-v2"))

    x = tf.convert_to_tensor([[0.2, 0.3], [0.4, 0.3], [0.2, 0.1]], dtype=tf.float32)
    alphas = tf.convert_to_tensor([[2, 1], [1, 3], [2, 2]], dtype=tf.float32)
    betas = tf.convert_to_tensor([[2, 2], [1, 2], [2, 1]], dtype=tf.float32)

    result_reference = np.prod(beta.pdf(distro._scale_sample_to_distribution_range(x), alphas, betas), axis=-1)
    result_pdf = distro.probability(x, alphas, betas).numpy()
    result_log_pdf = np.exp(distro.log_probability(x, alphas, betas).numpy())

    assert np.allclose(result_reference, result_log_pdf), "Beta Log PDF returns wrong Result"
    assert np.allclose(result_reference, result_pdf), "Beta PDF returns wrong Result"


def test_beta_entropy():
    """Beta entropy matches the SciPy reference.

    Property
        The summed per-dimension ``scipy.stats.beta.entropy(alpha, beta)`` equals
        the distribution's ``_entropy_from_params``.

    Rationale
        Validates the Beta entropy (exploration bonus) against an independent
        implementation, guarding the digamma/log-Beta terms that are easy to
        transcribe incorrectly.
    """
    distro = BetaPolicyDistribution(gym.make("LunarLanderContinuous-v2"))

    alphas = tf.convert_to_tensor([[2, 1], [1, 3], [2, 2]], dtype=tf.float32)
    betas = tf.convert_to_tensor([[2, 2], [1, 2], [2, 1]], dtype=tf.float32)

    result_reference = np.sum(beta.entropy(alphas, betas), axis=-1)
    result_pdf = distro._entropy_from_params((alphas, betas)).numpy()

    assert np.allclose(result_reference, result_pdf), "Beta PDF returns wrong Result"


# CATEGORICAL
def test_categorical_entropy():
    """Categorical entropy matches SciPy from both pmf and log-pmf.

    Property
        Per-row ``scipy.stats.entropy(p)`` equals the entropy computed from the
        raw pmf and from ``log(pmf)``.

    Rationale
        The discrete policy stores probabilities in log space; this pins both the
        entropy formula and the log/linear equivalence used in the exploration
        bonus for discrete control.
    """
    distro = CategoricalPolicyDistribution(gym.make("CartPole-v1"))

    probs = tf.convert_to_tensor([[0.1, 0.4, 0.2, 0.25, 0.05],
                                  [0.1, 0.4, 0.2, 0.2, 0.1],
                                  [0.1, 0.35, 0.3, 0.24, 0.01]], dtype=tf.float32)

    result_reference = [entropy(probs[i]) for i in range(len(probs))]
    result_log = distro._entropy_from_log_pmf(np.log(probs)).numpy()
    result = distro._entropy_from_pmf(probs).numpy()

    assert np.allclose(result_reference, result), "Discrete entropy returns wrong result"
    assert np.allclose(result_log, result_reference), "Discrete entropy from log returns wrong result"


# MULTI-CATEGORICAL
def test_multi_categorical_entropy():
    """Multi-categorical entropy matches SciPy per action dimension, from pmf and log-pmf.

    Property
        For a ``[batch, action_dim, categories]`` pmf, the per-dimension entropy
        equals ``scipy.stats.entropy`` over the categories axis, computed both
        from the raw pmf and from its log.

    Rationale
        The multi-categorical distribution drives the discrete ShadowHand
        manipulation tasks but is otherwise untested here. It treats action
        dimensions as independent categoricals; this verifies the per-dimension
        entropy (the exploration-bonus ingredient) is correct in both spaces.
    """
    distro = MultiCategoricalPolicyDistribution(_MultiDiscreteEnv())

    pmf = tf.convert_to_tensor([[[0.1, 0.4, 0.5], [0.2, 0.3, 0.5]],
                                [[0.6, 0.3, 0.1], [0.25, 0.25, 0.5]]], dtype=tf.float32)

    result_reference = [[entropy(pmf[b, d]) for d in range(pmf.shape[1])] for b in range(pmf.shape[0])]
    result = distro._entropy_from_pmf(pmf).numpy()
    result_log = distro._entropy_from_log_pmf(np.log(pmf)).numpy()

    assert np.allclose(result, result_reference), "Multi-categorical entropy returns wrong result"
    assert np.allclose(result_log, result_reference), "Multi-categorical entropy from log returns wrong result"


def test_categorical_tf_act_handles_recurrent_input():
    """``tf_act`` accepts a rank-3 (recurrent, single-timestep) input without error.

    Property
        Calling the ``tf.function``-compiled ``tf_act`` on a
        ``[batch, 1, n_actions]`` log-pmf returns a valid action index for the
        single timestep, exercising the sequence-dimension squeeze path.

    Rationale
        Regression test for the static-rank fix in ``tf_sample``: ``tf.rank`` is a
        dynamic tensor, which made autograph compile the rank check into a
        ``tf.cond`` that traced *both* branches and spuriously tripped the
        "single timestep" assertion. Using ``len(shape)`` keeps it a Python
        conditional; this guards that the recurrent path stays traceable.
    """
    distro = CategoricalPolicyDistribution(gym.make("CartPole-v1"))  # 2 discrete actions

    log_probabilities = tf.math.log(tf.constant([[[0.25, 0.75]]], dtype=tf.float32))  # (batch=1, seq=1, actions=2)
    action, _ = distro.tf_act(log_probabilities)

    assert int(action.numpy()) in (0, 1)


def test_extract_discrete_action_probabilities():
    """Per-sample gather selects the taken action's probability (feed-forward case).

    Property
        For a ``[batch, n_actions]`` prediction tensor and a ``[batch]`` action
        index vector, the result is ``predictions[i, actions[i]]``.

    Rationale
        This gather turns a full action distribution into the single
        log-probability used in the PPO ratio; an indexing error would feed the
        wrong probabilities into the objective.
    """
    # no recurrence
    action_probs = tf.convert_to_tensor([[1, 5], [3, 7], [7, 2], [8, 4], [0, 2], [4, 5], [4, 2], [7, 5]])
    actions = tf.convert_to_tensor([1, 0, 1, 1, 0, 0, 0, 1])
    result_reference = tf.convert_to_tensor([5, 3, 2, 4, 0, 4, 4, 5])
    result = extract_discrete_action_probabilities(action_probs, actions)

    assert tf.reduce_all(tf.equal(result, result_reference)).numpy().item()


def test_extract_discrete_action_probabilities_with_recurrence():
    """Per-sample gather also works for batched sequences (recurrent case).

    Property
        For a ``[batch, time, n_actions]`` prediction tensor and a
        ``[batch, time]`` action tensor, the result gathers each (batch, time)
        entry's taken-action probability, preserving the ``[batch, time]`` shape.

    Rationale
        Recurrent rollouts carry an extra time axis; this confirms the gather
        indexes the right axes and keeps the sequence layout intact. Functions
        are run eagerly here so the dynamic shapes are exercised directly.
    """
    tf.config.experimental_run_functions_eagerly(True)

    # with recurrence
    action_probs = tf.convert_to_tensor(
        [[[1, 5], [1, 5]], [[3, 7], [3, 7]], [[7, 2], [7, 2]], [[8, 4], [8, 4]], [[0, 2], [0, 2]], [[4, 5], [4, 5]],
         [[4, 2], [4, 2]], [[7, 5], [7, 5]]])
    actions = tf.convert_to_tensor([[1, 1], [0, 0], [1, 1], [1, 1], [0, 0], [0, 0], [0, 0], [1, 1]])
    result_reference = tf.convert_to_tensor([[5, 5], [3, 3], [2, 2], [4, 4], [0, 0], [4, 4], [4, 4], [5, 5]])
    result = extract_discrete_action_probabilities(action_probs, actions)

    assert tf.reduce_all(tf.equal(result, result_reference)).numpy().item()
