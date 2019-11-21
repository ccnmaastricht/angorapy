import os
import unittest

import numpy
import tensorflow as tf
from scipy.stats import norm, entropy

from agent.core import extract_discrete_action_probabilities, gaussian_pdf, gaussian_entropy, categorical_entropy


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


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.config.experimental_run_functions_eagerly(True)

    unittest.main()
