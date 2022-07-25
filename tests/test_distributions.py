import os

from angorapy.agent.core import extract_discrete_action_probabilities

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import unittest

import gym
import numpy as np
import tensorflow as tf
from scipy.stats import norm, entropy, beta

from angorapy.common.policies import GaussianPolicyDistribution, CategoricalPolicyDistribution, BetaPolicyDistribution


class ProbabilityTest(unittest.TestCase):

    # GAUSSIAN

    def test_gaussian_pdf(self):
        distro = GaussianPolicyDistribution(gym.make("LunarLanderContinuous-v2"))

        x = tf.convert_to_tensor([[2, 3], [4, 3], [2, 1]], dtype=tf.float32)
        mu = tf.convert_to_tensor([[2, 1], [1, 3], [2, 2]], dtype=tf.float32)
        sig = tf.convert_to_tensor([[2, 2], [1, 2], [2, 1]], dtype=tf.float32)

        result_reference = np.prod(norm.pdf(x, loc=mu, scale=sig), axis=-1)
        result_pdf = distro.probability(x, mu, sig).numpy()
        result_log_pdf = np.exp(distro.log_probability(x, mu, np.log(sig)).numpy())

        self.assertTrue(np.allclose(result_reference, result_pdf), msg="Gaussian PDF returns wrong Result")
        self.assertTrue(np.allclose(result_pdf, result_log_pdf), msg="Gaussian Log PDF returns wrong Result")

    def test_gaussian_entropy(self):
        distro = GaussianPolicyDistribution(gym.make("LunarLanderContinuous-v2"))

        mu = tf.convert_to_tensor([[2.0, 3.0], [2.0, 1.0]], dtype=tf.float32)
        sig = tf.convert_to_tensor([[1.0, 1.0], [1.0, 5.0]], dtype=tf.float32)

        result_reference = np.sum(norm.entropy(loc=mu, scale=sig), axis=-1)
        result_log = distro.entropy([mu, np.log(sig)]).numpy()
        result = distro._entropy_from_params(sig).numpy()

        self.assertTrue(np.allclose(result_reference, result), msg="Gaussian entropy returns wrong result")
        self.assertTrue(np.allclose(result_log, result_reference), msg="Gaussian entropy from log returns wrong result")

    # BETA

    def test_beta_pdf(self):
        distro = BetaPolicyDistribution(gym.make("LunarLanderContinuous-v2"))

        x = tf.convert_to_tensor([[0.2, 0.3], [0.4, 0.3], [0.2, 0.1]], dtype=tf.float32)
        alphas = tf.convert_to_tensor([[2, 1], [1, 3], [2, 2]], dtype=tf.float32)
        betas = tf.convert_to_tensor([[2, 2], [1, 2], [2, 1]], dtype=tf.float32)

        result_reference = np.prod(beta.pdf(distro._scale_sample_to_distribution_range(x), alphas, betas), axis=-1)
        result_pdf = distro.probability(x, alphas, betas).numpy()
        result_log_pdf = np.exp(distro.log_probability(x, alphas, betas).numpy())

        self.assertTrue(np.allclose(result_reference, result_log_pdf), msg="Beta Log PDF returns wrong Result")
        self.assertTrue(np.allclose(result_reference, result_pdf), msg="Beta PDF returns wrong Result")

    def test_beta_entropy(self):
        distro = BetaPolicyDistribution(gym.make("LunarLanderContinuous-v2"))

        alphas = tf.convert_to_tensor([[2, 1], [1, 3], [2, 2]], dtype=tf.float32)
        betas = tf.convert_to_tensor([[2, 2], [1, 2], [2, 1]], dtype=tf.float32)

        result_reference = np.sum(beta.entropy(alphas, betas), axis=-1)
        result_pdf = distro._entropy_from_params((alphas, betas)).numpy()

        self.assertTrue(np.allclose(result_reference, result_pdf), msg="Beta PDF returns wrong Result")

    # CATEGORICAL

    def test_categorical_entropy(self):
        distro = CategoricalPolicyDistribution(gym.make("CartPole-v1"))

        probs = tf.convert_to_tensor([[0.1, 0.4, 0.2, 0.25, 0.05],
                                      [0.1, 0.4, 0.2, 0.2, 0.1],
                                      [0.1, 0.35, 0.3, 0.24, 0.01]], dtype=tf.float32)

        result_reference = [entropy(probs[i]) for i in range(len(probs))]
        result_log = distro._entropy_from_log_pmf(np.log(probs)).numpy()
        result = distro._entropy_from_pmf(probs).numpy()

        self.assertTrue(np.allclose(result_reference, result), msg="Discrete entropy returns wrong result")
        self.assertTrue(np.allclose(result_log, result_reference), msg="Discrete entropy from log returns wrong result")

    def test_extract_discrete_action_probabilities(self):
        # no recurrence
        action_probs = tf.convert_to_tensor([[1, 5], [3, 7], [7, 2], [8, 4], [0, 2], [4, 5], [4, 2], [7, 5]])
        actions = tf.convert_to_tensor([1, 0, 1, 1, 0, 0, 0, 1])
        result_reference = tf.convert_to_tensor([5, 3, 2, 4, 0, 4, 4, 5])
        result = extract_discrete_action_probabilities(action_probs, actions)

        self.assertTrue(tf.reduce_all(tf.equal(result, result_reference)).numpy().item())

    def test_extract_discrete_action_probabilities_with_recurrence(self):
        tf.config.experimental_run_functions_eagerly(True)

        # with recurrence
        action_probs = tf.convert_to_tensor(
            [[[1, 5], [1, 5]], [[3, 7], [3, 7]], [[7, 2], [7, 2]], [[8, 4], [8, 4]], [[0, 2], [0, 2]], [[4, 5], [4, 5]],
             [[4, 2], [4, 2]], [[7, 5], [7, 5]]])
        actions = tf.convert_to_tensor([[1, 1], [0, 0], [1, 1], [1, 1], [0, 0], [0, 0], [0, 0], [1, 1]])
        result_reference = tf.convert_to_tensor([[5, 5], [3, 3], [2, 2], [4, 4], [0, 0], [4, 4], [4, 4], [5, 5]])
        result = extract_discrete_action_probabilities(action_probs, actions)

        self.assertTrue(tf.reduce_all(tf.equal(result, result_reference)).numpy().item())