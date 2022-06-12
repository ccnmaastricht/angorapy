import unittest

import numpy as np

from angorapy.utilities.statistics import increment_mean_var


class StatisticsTest(unittest.TestCase):

    def test_incremental_mean_var(self):
        n_samples = 100000
        sample_dims = 10

        samples = np.array([np.random.randn(1, sample_dims) for _ in range(n_samples)])

        mean, var, n = samples[0], np.zeros((1, sample_dims, )), 1
        for s in samples[1:]:
            s_var = np.zeros((sample_dims,))
            mean, var = increment_mean_var(mean, var, s, s_var, n)
            n += 1

        np_mean = np.mean(samples, axis=0)
        np_var = np.var(samples, axis=0)

        self.assertTrue(np.allclose(mean, np_mean),
                        np.allclose(mean, np_var))