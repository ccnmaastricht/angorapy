import os
import unittest

import tensorflow as tf

from agent.core import extract_discrete_action_probabilities


class CoreTest(unittest.TestCase):

    def test_extract_discrete_action_probabilities(self):
        action_probs = tf.convert_to_tensor([[1, 5], [3, 7], [7, 2], [8, 4], [0, 2], [4, 5], [4, 2], [7, 5]])
        actions = tf.convert_to_tensor([1, 0, 1, 1, 0, 0, 0, 1])
        result_reference = tf.convert_to_tensor([5, 3, 2, 4, 0, 4, 4, 5])

        result = extract_discrete_action_probabilities(action_probs, actions)

        self.assertTrue(tf.reduce_all(tf.equal(result, result_reference)).numpy().item())


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.config.experimental_run_functions_eagerly(True)

    unittest.main()
