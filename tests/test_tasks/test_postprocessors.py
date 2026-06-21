"""Unit tests for the running-statistics postprocessors in :mod:`angorapy.common.postprocessors`.

:class:`StateNormalizer` and :class:`RewardNormalizer` maintain online estimates
of the mean and variance of observations / rewards and use them to standardize the
data stream during training. Two properties are essential and tested here:

1. **Correctness of the online estimate** — after consuming a stream, the
   normalizer's running mean/variance must match the exact batch statistics of
   that stream (for both dict-structured ``Sensation`` observations and plain
   array observations).
2. **Correct parallel combination** — normalizers accumulated independently
   (e.g. on separate MPI workers) must combine, via ``+`` and
   :func:`merge_postprocessors`, into the same statistics as if one normalizer had
   seen all the data. This is what makes distributed statistic-gathering valid.
"""
import random
import unittest

import numpy as np

from angorapy.tasks.registration import make_task
from angorapy.common.senses import Sensation
from angorapy.common.postprocessors import RewardNormalizer, merge_postprocessors, \
    StateNormalizer, postprocessors_from_serializations
from angorapy.utilities.core import env_extract_dims


def test_state_normalization():
    """State normalizer's running moments match the batch statistics (dict observations).

    Property
        After transforming 150 stepped observations from a dict-observation
        (ShadowHand) task, the normalizer's per-sense ``mean`` and ``variance``
        equal the directly-computed batch mean and variance.

    Rationale
        Validates the online update on the structured ``Sensation`` observation
        type, where statistics must be tracked independently per sense.
    """
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
    """State normalizer's running moments match batch statistics (plain box observations).

    Property
        Same as above but for a flat ``Box`` observation task (LunarLander):
        running mean/variance equal the batch statistics over 150 steps.

    Rationale
        Confirms the normalizer handles the non-dict (single ``proprioception``)
        observation case as well as the structured one.
    """
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
    """Independently-updated state normalizers combine to the pooled statistics.

    Property
        Three normalizers each fed a disjoint third of the data, then combined via
        ``+`` and via :func:`merge_postprocessors`, both yield the mean (and std)
        of the full concatenated data set.

    Rationale
        Distributed training accumulates statistics per worker and merges them;
        this verifies the parallel mean/variance combination is exact, so the
        merged normalizer is identical to a single-process one.
    """
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
    """Independently-updated reward normalizers combine to the pooled statistics.

    Property
        Three reward normalizers fed disjoint streams (with different scales),
        combined via ``+``, yield the mean and std of the concatenated rewards.

    Rationale
        The reward counterpart to the state-normalizer merge test; ensures scalar
        reward statistics also combine exactly across workers.
    """
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


def test_state_normalizer_serialization_roundtrip():
    """A normalizer survives a serialize -> recover round-trip with identical statistics.

    Property
        After updating a :class:`StateNormalizer`, ``serialize()`` followed by
        :func:`postprocessors_from_serializations` reproduces the same sample
        count, mean, and variance.

    Rationale
        Normalizers are persisted alongside a trained agent and restored when it
        is reloaded/evaluated, via exactly this path. If the round-trip drops or
        corrupts the running statistics, a resumed agent silently normalizes its
        observations with the wrong moments — this guards that contract.
    """
    env_name = "LunarLanderContinuous-v2"
    env = make_task(env_name)
    normalizer = StateNormalizer(env_name, *env_extract_dims(env))

    for _ in range(100):
        normalizer.update({"proprioception": env.observation_space.sample()})

    recovered = postprocessors_from_serializations([normalizer.serialize()])[0]

    assert recovered.n == normalizer.n
    assert np.allclose(recovered.mean["proprioception"], normalizer.mean["proprioception"])
    assert np.allclose(recovered.variance["proprioception"], normalizer.variance["proprioception"])
