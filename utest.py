import os
import unittest

import numpy
import tensorflow as tf
from scipy.signal import lfilter
from scipy.stats import norm, entropy

from agent.core import extract_discrete_action_probabilities, gaussian_pdf, gaussian_entropy, categorical_entropy, \
    estimate_advantage, estimate_episode_advantages

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class CoreTest(unittest.TestCase):

    def test_extract_discrete_action_probabilities(self):
        action_probs = tf.convert_to_tensor([[1, 5], [3, 7], [7, 2], [8, 4], [0, 2], [4, 5], [4, 2], [7, 5]])
        actions = tf.convert_to_tensor([1, 0, 1, 1, 0, 0, 0, 1])
        result_reference = tf.convert_to_tensor([5, 3, 2, 4, 0, 4, 4, 5])

        result = extract_discrete_action_probabilities(action_probs, actions)

        self.assertTrue(tf.reduce_all(tf.equal(result, result_reference)).numpy().item())


class ProbabilityTest(unittest.TestCase):

    def test_gaussian_pdf(self):
        x = tf.convert_to_tensor([[2, 3]], dtype=tf.float32)
        mu = tf.convert_to_tensor([[2, 3]], dtype=tf.float32)
        sig = tf.convert_to_tensor([[1, 1]], dtype=tf.float32)

        result_reference = numpy.prod(norm.pdf(x, loc=mu, scale=sig))
        result = gaussian_pdf(x, mu, sig).numpy().item()

        self.assertTrue(numpy.allclose(result_reference, result))

    def test_gaussian_entropy(self):
        mu = tf.convert_to_tensor([[2, 3]], dtype=tf.float32)
        sig = tf.convert_to_tensor([[1, 1]], dtype=tf.float32)

        result_reference = numpy.sum(norm.entropy(loc=mu, scale=sig))
        result = gaussian_entropy(sig).numpy().item()

        self.assertTrue(numpy.allclose(result_reference, result))

    def test_categorical_entropy(self):
        probs = tf.convert_to_tensor([[0.1, 0.4, 0.2, 0.25, 0.05]], dtype=tf.float32)

        result_reference = entropy(probs[0].numpy())
        result = categorical_entropy(probs).numpy().item()



        self.assertTrue(numpy.allclose(result_reference, result))


class AdvantageTest(unittest.TestCase):

    def test_gae_estimation(self):
        def discount(x, gamma):
            return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

        rewards = numpy.array([5, 7, 9, 2, 9, 5, 7, 3, 7, 8, 4.0, 10, 10, 4, 9, 4])
        values = numpy.array([9, 7, 7, 5, 6, 8, 2, 2, 5, 3, 3.0, 4, 10, 5, 7, 8, 9])
        terminals = numpy.array(
            [False, False, False, False, False, False, False, False, False, False, True,
             False, False, False, False, False])

        rewards = numpy.array([5, 7, 9, 2, 9, 5, 7, 3, 7, 8, 4.0])
        values = numpy.array([9, 7, 7, 5, 6, 8, 2, 2, 5, 3, 3.0, 0])
        terminals = numpy.array([False, False, False, False, False, False, False, False, False, False, True])

        gamma = 1
        lamb = 1

        result = estimate_advantage(rewards, values, terminals, gamma=gamma, lam=lamb)

        prep = rewards + gamma * values[1:] * (1 - terminals) - values[:-1]
        result_reference = discount(prep, gamma * lamb)

        print(result)
        print(result_reference)
        print(estimate_episode_advantages(rewards, values, gamma, lamb))


if __name__ == '__main__':
    tf.config.experimental_run_functions_eagerly(True)

    unittest.main()
