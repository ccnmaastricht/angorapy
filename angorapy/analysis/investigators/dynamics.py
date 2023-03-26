import numpy as np
import sklearn.model_selection
import tqdm

from angorapy.analysis.investigators import base_investigator

import tensorflow as tf

from angorapy.analysis.sindy.autoencoder import SindyAutoencoder
from angorapy.analysis.util.sindy import compute_z_derivatives, sindy_library_tf
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
    """Discover dynamics in latent variables using the SINDy autoencoder algorithm."""

    def __init__(self, network: tf.keras.Model, distribution: BasePolicyDistribution):
        super().__init__(network, distribution)

        self.loss_weights = {
            "x": 5e-4,
            "y": 5e-5,
            "regularization": 1e-5
        }

    def fit(self, n_epochs: int = 10, batch_size: int = 128):
        data = tf.squeeze(self.train_data)

        sindy_autoencoder = SindyAutoencoder([256, 128, 32], z_dim=32, original_dim=data.shape[-1])
        optimizer = tf.keras.optimizers.Adam()

        dataset = tf.data.Dataset.from_tensor_slices((data, data))
        dataset = dataset.batch(batch_size)

        for epoch in range(n_epochs):

            total_epoch_loss = 0
            for step, (x_train_batch, y_train_batch) in enumerate(dataset):

                with tf.GradientTape() as tape:
                    z_coordinates, reconstruction, dz_prediction = sindy_autoencoder(x_train_batch)

                    dx = np.gradient(x_train_batch, axis=1)  # todo axis=0 or 1?
                    dz = compute_z_derivatives(
                        x_train_batch,
                        dx,
                        weights=[layer.get_weights()[0] for layer in sindy_autoencoder.encoder.layers],
                        biases=[layer.get_weights()[1] for layer in sindy_autoencoder.encoder.layers]
                    )
                    dx_decoded = compute_z_derivatives(
                        z_coordinates,
                        dz_prediction,
                        weights=[layer.get_weights()[0] for layer in sindy_autoencoder.decoder.layers],
                        biases=[layer.get_weights()[1] for layer in sindy_autoencoder.decoder.layers]
                    )

                    reconstruction_loss = tf.reduce_mean((x_train_batch - reconstruction) ** 2)
                    sindy_x_loss = tf.reduce_mean((dx - dx_decoded) ** 2)
                    sindy_z_loss = tf.reduce_mean((dz - dz_prediction) ** 2)
                    sindy_regularization_loss = tf.reduce_mean(tf.abs(sindy_autoencoder.coefficients))

                    loss_value = reconstruction_loss \
                                 + sindy_x_loss * self.loss_weights["x"] \
                                 + sindy_z_loss * self.loss_weights["y"] \
                                 + sindy_regularization_loss * self.loss_weights["regularization"]

                grads = tape.gradient(loss_value, sindy_autoencoder.trainable_weights)
                optimizer.apply_gradients(zip(grads, sindy_autoencoder.trainable_weights))

                total_epoch_loss += float(loss_value)

            print(f"Current loss after {epoch} epochs: {total_epoch_loss}")

