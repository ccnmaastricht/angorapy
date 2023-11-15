import random
import unittest

import numpy as np

from angorapy.tasks.registration import make_task
from angorapy.common.senses import Sensation
from angorapy.common.postprocessors import RewardNormalizer, merge_postprocessors, \
    StateNormalizer
from angorapy.utilities.core import env_extract_dims


def test_state_normalization():
    env_name = "ManipulateBlockDiscreteAsynchronous-v0"
    env = make_task(env_name)
    normalizer = StateNormalizer(env_name, *env_extract_dims(env))

    env.reset()
    inputs = [env.step(env.action_space.sample())[0] for _ in range(150)]

    for sample in inputs:
        o, _, _, _, _ = normalizer.transform((sample, 1.0, False, False, {}))

    true_mean = np.sum(inputs, axis=0) / len(inputs)
    true_std = np.mean([(i - Sensation(**normalizer.mean)) ** 2 for i in inputs])

    for name in true_mean.dict().keys():
        assert np.allclose(true_mean[name], normalizer.mean[name], atol=1e-6), f"{name}'s mean not equal."
        assert np.allclose(true_std[name], normalizer.variance[name], atol=1e-5), f"{name}'s std not equal."


def test_state_normalization_non_anthropomorphic():
    env_name = "LunarLanderContinuous-v2"
    env = make_task(env_name)
    normalizer = StateNormalizer(env_name, *env_extract_dims(env))

    env.reset()
    inputs = [env.step(env.action_space.sample())[0] for _ in range(150)]
    true_mean = np.sum(inputs, axis=0) / len(inputs)
    true_std = np.mean([(i - true_mean) ** 2 for i in inputs])

    for sample in inputs:
        o, _, _, _, _ = normalizer.transform((sample, 1.0, False, False, {}))

    for name in true_mean.dict().keys():
        assert np.allclose(true_mean[name], normalizer.mean[name], atol=1e-6), f"{name}'s mean not equal."
        assert np.allclose(true_std[name], normalizer.variance[name], atol=1e-5), f"{name}'s std not equal."


def test_state_normalization_adding():
    env_name = "LunarLanderContinuous-v2"
    env = make_task(env_name)
    normalizer_a = StateNormalizer(env_name, *env_extract_dims(env))
    normalizer_b = StateNormalizer(env_name, *env_extract_dims(env))
    normalizer_c = StateNormalizer(env_name, *env_extract_dims(env))

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
    merged_normalizer = merge_postprocessors([normalizer_a, normalizer_b, normalizer_c])

    assert np.allclose(true_mean, combined_normalizer.mean["proprioception"], atol=1e-6)
    assert np.allclose(true_mean, merged_normalizer.mean["proprioception"], atol=1e-6)
    assert np.allclose(true_std, np.sqrt(combined_normalizer.variance["proprioception"]), atol=1e-6)


def test_reward_normalization_adding():
    env_name = "LunarLanderContinuous-v2"
    env = make_task(env_name)
    normalizer_a = RewardNormalizer(env_name, *env_extract_dims(env))
    normalizer_b = RewardNormalizer(env_name, *env_extract_dims(env))
    normalizer_c = RewardNormalizer(env_name, *env_extract_dims(env))

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

    assert np.allclose(true_mean, combined_normalizer.mean["reward"])
    assert np.allclose(true_std, np.sqrt(combined_normalizer.variance["reward"]))
