import os
import sys
from typing import List

import numpy as np
import tensorflow as tf
import tqdm
from angorapy.utilities.core import flatten
from angorapy.utilities.core import stack_dicts
from matplotlib import pyplot as plt

from angorapy.analysis.investigators import base_investigator
from angorapy.common.policies import BasePolicyDistribution
from angorapy.tasks.wrappers import TaskWrapper


class ActivityAutocorrelation(base_investigator.Investigator):
    """Investigate autocorrelation of the activity in different regions of the network as a measure of how long
    information is maintained over time."""

    def __init__(self, network: tf.keras.Model, distribution: BasePolicyDistribution):
        super().__init__(network, distribution)
        self._data = None
        self.prepared = False

        self.n_states = np.nan

    def prepare(self, env: TaskWrapper, layers: List[str], n_states=1000, n_repeats=1, verbose=True):
        """Prepare samples to predict from."""
        self.network.reset_states()
        state, info = env.reset(return_info=True)

        repeated_collections = []
        for i in range(n_repeats):
            activation_collection = []
            for _ in tqdm.tqdm(range(n_states), disable=not verbose):
                prepared_state = state.with_leading_dims(time=self.is_recurrent).dict()

                # make step and record activities
                activations = self.get_layer_activations(layers, prepared_state)
                probabilities = flatten(activations["output"])
                activations.pop("output")
                activation_collection.append(activations)

                action, _ = self.distribution.act(*probabilities)

                observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if done:
                    state, info = env.reset(return_info=True)
                    self.network.reset_states()
                else:
                    state = observation

            repeated_collections.append(activation_collection)

        self._data = stack_dicts([stack_dicts(ac) for ac in repeated_collections])
        self.prepared = True

    def fit(self, n_lags=30):
        """Measure the predictability of target_information based on the information in source_layer's activation."""
        assert self.prepared, "Need to prepare before investigating."

        # calculate autocorrelation for each layer
        autocorrelation = {}
        for layer in self._data.keys():
            layer_autocorrelations = []
            for repeat in range(self._data[layer].shape[0]):
                layer_autocorrelations.append(self.calculate_statewise_correlation(
                    np.squeeze(self._data[layer][repeat]), n_lags=n_lags)
                )

            autocorrelation[layer] = np.mean(layer_autocorrelations, axis=0)

        return autocorrelation

    def calculate_statewise_correlation(self, data, n_lags=30):
        corr_coeffs = np.corrcoef(data)
        lag_shifted_corr_coeffs = np.zeros_like(corr_coeffs)
        for row in range(n_lags):
            lag_shifted_corr_coeffs[row] = np.roll(corr_coeffs[row], -row)

        return lag_shifted_corr_coeffs[:n_lags, :n_lags]
