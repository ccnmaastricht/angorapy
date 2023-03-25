import random
import unittest

import numpy as np

from angorapy import make_env
from angorapy.common.const import NP_FLOAT_PREC
from angorapy.common.senses import Sensation
from angorapy.common.transformers import RewardNormalizationTransformer, merge_transformers, \
    StateNormalizationTransformer
from angorapy.utilities.util import env_extract_dims


class EnvironmentTest(unittest.TestCase):

    def test_state_normalization(self):
        env_name = "HumanoidManipulateBlockDiscreteAsynchronous-v0"
        env = make_env(env_name)
        normalizer = StateNormalizationTransformer(env_name, *env_extract_dims(env))

        env.reset()
        inputs = [env.step(env.action_space.sample())[0] for _ in range(150)]

        for sample in inputs:
            o, _, _, _, _ = normalizer.transform((sample, 1.0, False, False, {}))

        true_mean = np.sum(inputs, axis=0) / len(inputs)
        true_std = np.mean([(i - Sensation(**normalizer.mean))**2 for i in inputs])

        for name in true_mean.dict().keys():
            self.assertTrue(np.allclose(true_mean[name], normalizer.mean[name]), msg=f"{name}'s mean not equal.")
            self.assertTrue(np.allclose(true_std[name], normalizer.variance[name], atol=1e-5), msg=f"{name}'s std not equal.")

    def test_state_normalization_non_anthropomorphic(self):
        env_name = "LunarLanderContinuous-v2"
        env = make_env(env_name)
        normalizer = StateNormalizationTransformer(env_name, *env_extract_dims(env))

        env.reset()
        inputs = [env.step(env.action_space.sample())[0] for _ in range(150)]
        true_mean = np.sum(inputs, axis=0) / len(inputs)
        true_std = np.mean([(i - true_mean)**2 for i in inputs])

        for sample in inputs:
            o, _, _, _, _ = normalizer.transform((sample, 1.0, False, False, {}))

        for name in true_mean.dict().keys():
            self.assertTrue(np.allclose(true_mean[name], normalizer.mean[name]), msg=f"{name}'s mean not equal.")
            self.assertTrue(np.allclose(true_std[name], normalizer.variance[name], atol=1e-5), msg=f"{name}'s std not equal.")

    def test_reward_normalization(self):
        # testing not so straight forward because its based on the return, maybe later todo
        pass

    def test_state_normalization_adding(self):
        env_name = "LunarLanderContinuous-v2"
        env = make_env(env_name)
        normalizer_a = StateNormalizationTransformer(env_name, *env_extract_dims(env))
        normalizer_b = StateNormalizationTransformer(env_name, *env_extract_dims(env))
        normalizer_c = StateNormalizationTransformer(env_name, *env_extract_dims(env))

        inputs_a = [env.observation_space.sample() for _ in range(100)]
        inputs_b = [env.observation_space.sample() for _ in range(100)]
        inputs_c = [env.observation_space.sample() for _ in range(100)]

        true_mean = np.mean(inputs_a + inputs_b + inputs_c, axis=0)
        true_std = np.std(inputs_a + inputs_b + inputs_c, axis=0)

        for sample in inputs_a:
            normalizer_a.update({"proprioception": sample})

        for sample in inputs_b:
            normalizer_b.update({"proprioception": sample})

        for sample in inputs_c:
            normalizer_c.update({"proprioception": sample})

        combined_normalizer = normalizer_a + normalizer_b + normalizer_c
        merged_normalizer = merge_transformers([normalizer_a, normalizer_b, normalizer_c])

        # print(normalizer_a.mean["proprioception"])
        # print(normalizer_b.mean["proprioception"])
        # print(normalizer_c.mean["proprioception"])
        # print(true_mean)
        # print(combined_normalizer.mean["proprioception"])

        self.assertTrue(np.allclose(true_mean, combined_normalizer.mean["proprioception"]))
        self.assertTrue(np.allclose(true_mean, merged_normalizer.mean["proprioception"]))
        self.assertTrue(np.allclose(true_std, np.sqrt(combined_normalizer.variance["proprioception"])))

    def test_reward_normalization_adding(self):
        env_name = "LunarLanderContinuous-v2"
        env = make_env(env_name)
        normalizer_a = RewardNormalizationTransformer(env_name, *env_extract_dims(env))
        normalizer_b = RewardNormalizationTransformer(env_name, *env_extract_dims(env))
        normalizer_c = RewardNormalizationTransformer(env_name, *env_extract_dims(env))

        inputs_a = [random.random() * 10 for _ in range(1000)]
        inputs_b = [random.random() * 20 for _ in range(1000)]
        inputs_c = [random.random() * 5 for _ in range(1000)]

        true_mean = np.mean(inputs_a + inputs_b + inputs_c, axis=0)
        true_std = np.std(inputs_a + inputs_b + inputs_c, axis=0)

        for sample in inputs_a:
            normalizer_a.update({"reward": sample})

        for sample in inputs_b:
            normalizer_b.update({"reward": sample})

        for sample in inputs_c:
            normalizer_c.update({"reward": sample})

        combined_normalizer = normalizer_a + normalizer_b + normalizer_c

        self.assertTrue(np.allclose(true_mean, combined_normalizer.mean["reward"]))
        self.assertTrue(np.allclose(true_std, np.sqrt(combined_normalizer.variance["reward"])))