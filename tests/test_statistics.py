import numpy as np

from angorapy.utilities.statistics import increment_mean_var


def test_incremental_mean_var():
    n_samples = 100000
    sample_dims = 10

    samples = np.array([np.random.randn(1, sample_dims) for _ in range(n_samples)])

    mean, var, n = samples[0], np.zeros((1, sample_dims,)), 1
    for s in samples[1:]:
        s_var = np.zeros((sample_dims,))
        mean, var = increment_mean_var(mean, var, s, s_var, n)
        n += 1

    np_mean = np.mean(samples, axis=0)
    np_var = np.var(samples, axis=0)

    assert np.allclose(mean, np_mean)
    assert np.allclose(var, np_var)
