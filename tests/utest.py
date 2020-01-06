import os
import random
import unittest

import gym
import numpy as np
import tensorflow as tf
from scipy.signal import lfilter
from scipy.stats import norm, entropy

from agent.core import extract_discrete_action_probabilities, estimate_advantage
from agent.policy import GaussianPolicyDistribution, CategoricalPolicyDistribution
from agent.ppo import PPOAgent
from models import get_model_builder
from utilities.util import reset_states_masked
from utilities.wrappers import StateNormalizationWrapper, RewardNormalizationWrapper

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class CoreTest(unittest.TestCase):

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


class ProbabilityTest(unittest.TestCase):

    def test_gaussian_pdf(self):
        distro = GaussianPolicyDistribution()

        x = tf.convert_to_tensor([[2, 3], [4, 3], [2, 1]], dtype=tf.float32)
        mu = tf.convert_to_tensor([[2, 1], [1, 3], [2, 2]], dtype=tf.float32)
        sig = tf.convert_to_tensor([[2, 2], [1, 2], [2, 1]], dtype=tf.float32)

        result_reference = np.prod(norm.pdf(x, loc=mu, scale=sig), axis=-1)
        result_pdf = distro.probability(x, mu, sig).numpy()
        result_log_pdf = np.exp(distro.log_probability(x, mu, np.log(sig)).numpy())

        self.assertTrue(np.allclose(result_reference, result_pdf), msg="Gaussian PDF returns wrong Result")
        self.assertTrue(np.allclose(result_pdf, result_log_pdf), msg="Gaussian Log PDF returns wrong Result")

    def test_gaussian_entropy(self):
        distro = GaussianPolicyDistribution()

        mu = tf.convert_to_tensor([[2.0, 3.0], [2.0, 1.0]], dtype=tf.float32)
        sig = tf.convert_to_tensor([[1.0, 1.0], [1.0, 5.0]], dtype=tf.float32)

        result_reference = np.sum(norm.entropy(loc=mu, scale=sig), axis=-1)
        result_log = distro.entropy_from_log(np.log(sig)).numpy()
        result = distro.entropy(sig).numpy()

        self.assertTrue(np.allclose(result_reference, result), msg="Gaussian entropy returns wrong result")
        self.assertTrue(np.allclose(result_log, result_reference), msg="Gaussian entropy from log returns wrong result")

    def test_categorical_entropy(self):
        distro = CategoricalPolicyDistribution()

        probs = tf.convert_to_tensor([[0.1, 0.4, 0.2, 0.25, 0.05],
                                      [0.1, 0.4, 0.2, 0.2, 0.1],
                                      [0.1, 0.35, 0.3, 0.24, 0.01]], dtype=tf.float32)

        result_reference = [entropy(probs[i]) for i in range(len(probs))]
        result_log = distro.entropy_from_log(np.log(probs)).numpy()
        result = distro.entropy(probs).numpy()

        self.assertTrue(np.allclose(result_reference, result), msg="Discrete entropy returns wrong result")
        self.assertTrue(np.allclose(result_log, result_reference), msg="Discrete entropy from log returns wrong result")


class AdvantageTest(unittest.TestCase):

    def test_gae_estimation(self):
        def discount(x, gamma):
            return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

        rewards = np.array([5, 7, 9, 2, 9, 5, 7, 3, 7, 8, 4.0, 10, 10, 4, 9, 4])
        values = np.array([9, 7, 7, 5, 6, 8, 2, 2, 5, 3, 3.0, 4, 10, 5, 7, 8, 9])
        terminals = np.array(
            [False, False, False, False, False, False, False, False, False, False, True,
             False, False, False, False, False])

        rewards = np.array([5, 7, 9, 2, 9, 5, 7, 3, 7, 8, 4.0])
        values = np.array([9, 7, 7, 5, 6, 8, 2, 2, 5, 3, 3.0, 0])
        terminals = np.array([False, False, False, False, False, False, False, False, False, False, True])

        gamma = 1
        lamb = 1

        result = estimate_advantage(rewards, values, terminals, gamma=gamma, lam=lamb)

        prep = rewards + gamma * values[1:] * (1 - terminals) - values[:-1]
        result_reference = discount(prep, gamma * lamb)


class UtilTest(unittest.TestCase):

    def test_masked_state_reset(self):
        model = tf.keras.Sequential((
            tf.keras.layers.Dense(2, batch_input_shape=(7, None, 2)),
            tf.keras.layers.LSTM(5, stateful=True, name="larry", return_sequences=True),
            tf.keras.layers.LSTM(5, stateful=True, name="harry"))
        )

        l_layer = model.get_layer("larry")
        h_layer = model.get_layer("harry")
        l_layer.reset_states([s.numpy() + 9 for s in l_layer.states])
        h_layer.reset_states([s.numpy() + 9 for s in h_layer.states])
        reset_states_masked(model, [True, False, False, True, False, False, True])

        self.assertTrue(np.allclose([s.numpy() for s in model.get_layer("larry").states],
                                    [s.numpy() for s in model.get_layer("harry").states]))
        self.assertTrue(np.allclose([s.numpy() for s in model.get_layer("larry").states], [
            [0, 0, 0, 0, 0],
            [9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9],
            [0, 0, 0, 0, 0],
            [9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9],
            [0, 0, 0, 0, 0],
        ]))


class GatheringTest(unittest.TestCase):

    def test_type_equivalence(self):
        """Test if recurrent and non-recurrent gathering both produce the same experience."""
        pass


class WrapperTest(unittest.TestCase):

    def test_state_normalization(self):
        normalizer = StateNormalizationWrapper(10)

        inputs = [tf.random.normal([10]) for _ in range(10)]
        true_mean = np.mean(inputs, axis=0)
        true_std = np.std(inputs, axis=0)

        for sample in inputs:
            o, _, _, _ = normalizer.wrap_a_step((sample, 1, 1, 1))

        self.assertTrue(np.allclose(true_mean, normalizer.mean))
        self.assertTrue(np.allclose(true_std, np.sqrt(normalizer.variance)))

    def test_reward_normalization(self):
        normalizer = RewardNormalizationWrapper()

        inputs = [random.random() * 10 for _ in range(10)]
        true_mean = np.mean(inputs, axis=0)
        true_std = np.std(inputs, axis=0)

        for sample in inputs:
            o, _, _, _ = normalizer.wrap_a_step((1, sample, 1, 1))

        self.assertTrue(np.allclose(true_mean, normalizer.mean))
        self.assertTrue(np.allclose(true_std, np.sqrt(normalizer.variance)))

    def test_state_normalization_adding(self):
        normalizer_a = StateNormalizationWrapper(10)
        normalizer_b = StateNormalizationWrapper(10)
        normalizer_c = StateNormalizationWrapper(10)

        inputs_a = [tf.random.normal([10]) for _ in range(10)]
        inputs_b = [tf.random.normal([10]) for _ in range(10)]
        inputs_c = [tf.random.normal([10]) for _ in range(10)]

        true_mean = np.mean(inputs_a + inputs_b + inputs_c, axis=0)
        true_std = np.std(inputs_a + inputs_b + inputs_c, axis=0)

        for sample in inputs_a:
            o, _, _, _ = normalizer_a.wrap_a_step((sample, 1, 1, 1))

        for sample in inputs_b:
            o, _, _, _ = normalizer_b.wrap_a_step((sample, 1, 1, 1))

        for sample in inputs_c:
            o, _, _, _ = normalizer_c.wrap_a_step((sample, 1, 1, 1))

        combined_normalizer = normalizer_a + normalizer_b + normalizer_c

        self.assertTrue(np.allclose(true_mean, combined_normalizer.mean))
        self.assertTrue(np.allclose(true_std, np.sqrt(combined_normalizer.variance)))

    def test_reward_normalization_adding(self):
        normalizer_a = RewardNormalizationWrapper()
        normalizer_b = RewardNormalizationWrapper()
        normalizer_c = RewardNormalizationWrapper()

        inputs_a = [random.random() * 10 for _ in range(10)]
        inputs_b = [random.random() * 20 for _ in range(10)]
        inputs_c = [random.random() * 5 for _ in range(10)]

        true_mean = np.mean(inputs_a + inputs_b + inputs_c, axis=0)
        true_std = np.std(inputs_a + inputs_b + inputs_c, axis=0)

        for sample in inputs_a:
            o, _, _, _ = normalizer_a.wrap_a_step((1, sample, 1, 1))

        for sample in inputs_b:
            o, _, _, _ = normalizer_b.wrap_a_step((1, sample, 1, 1))

        for sample in inputs_c:
            o, _, _, _ = normalizer_c.wrap_a_step((1, sample, 1, 1))

        combined_normalizer = normalizer_a + normalizer_b + normalizer_c

        self.assertTrue(np.allclose(true_mean, combined_normalizer.mean))
        self.assertTrue(np.allclose(true_std, np.sqrt(combined_normalizer.variance)))


class AgentTest(unittest.TestCase):

    def test_saving_loading(self):
        try:
            agent = PPOAgent(get_model_builder("ffn", True), gym.make("CartPole-v1"), 100, 8)
            agent.save_agent_state()
            new_agent = PPOAgent.from_agent_state(agent.agent_id)
        except Exception:
            self.assertTrue(False)


if __name__ == '__main__':
    tf.config.experimental_run_functions_eagerly(True)

    unittest.main()
