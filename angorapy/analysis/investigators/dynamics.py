import numpy as np
import sklearn.model_selection
import tqdm

from angorapy.analysis.investigators import base_investigator

import tensorflow as tf

from angorapy.common.policies import BasePolicyDistribution
from angorapy.common.wrappers import BaseWrapper
from angorapy.utilities import util

import pysindy as ps


class Dynamics(base_investigator.Investigator):
    """Investigate dynamics of the system constituted by a recurrent model."""

    def __init__(self, network: tf.keras.Model, distribution: BasePolicyDistribution):
        super().__init__(network, distribution)
        self._data = None
        self.is_prepared = False

        assert self.is_recurrent, "Network is not recurrent, no system to be identified without temporal dependencies."

    def prepare(self, env: BaseWrapper, layer: str, n_states: int):
        """Collect data for the system to be identified from via SINDy."""
        state, info = env.reset(return_info=True)

        activation_collection = []
        for _ in tqdm.tqdm(range(n_states)):
            prepared_state = state.with_leading_dims(time=self.is_recurrent).dict()
            activations = self.get_layer_activations([layer], prepared_state)
            activations[layer] = activations[layer][1]
            activation_collection.append(activations)

            probabilities = util.flatten(activations["output"])
            action, _ = self.distribution.act(*probabilities)

            observation, reward, done, info = env.step(action)

            if done:
                state, info = env.reset(return_info=True)
                self.network.reset_states()
            else:
                state = observation
        self._data = util.stack_dicts(activation_collection)[layer]
        self.train_data, self.val_data = sklearn.model_selection.train_test_split(self._data, shuffle=False, test_size=0.2)
        self.is_prepared = True

    def fit(self):
        """Fit a SINDy regressor to the prepared data."""
        assert self.is_prepared, "Need to prepare data first. Run .prepare(...)."

        model = ps.SINDy(
            optimizer=ps.STLSQ(threshold=0.0001),
            feature_library=ps.PolynomialLibrary(degree=2)
        )

        model.fit(self.train_data, t=0.1)
        model.print()


class LatentDynamics(Dynamics):

    def __init__(self, network: tf.keras.Model, distribution: BasePolicyDistribution):
        super().__init__(network, distribution)

    def fit(self):
        autoencoder = None
        optimizer = tf.keras.optimizers.Adam()

    def sindy_ae_loss(self):
        pass
