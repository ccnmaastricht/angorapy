from typing import List

import tensorflow as tf
import tqdm

from dexterity.analysis.investigators import base_investigator
from dexterity.common.policies import BasePolicyDistribution
from dexterity.common.senses import stack_sensations
from dexterity.common.wrappers import BaseWrapper
from dexterity.utilities.util import flatten, stack_dicts


class Predictability(base_investigator.Investigator):
    """Investigate predictability of information from the activity in different regions of the network."""

    def __init__(self, network: tf.keras.Model, distribution: BasePolicyDistribution):
        super().__init__(network, distribution)
        self._data = None
        self.prepared = False

    def prepare(self, env: BaseWrapper, layers: List[str]):
        """Prepare samples to predict from."""
        state = env.reset()
        state_collection = []
        activation_collection = []
        n_states = 10000
        for _ in tqdm.tqdm(range(n_states)):
            state_collection.append(state)

            prepared_state = state.with_leading_dims(time=self.is_recurrent).dict()
            activation_collection.append(self.get_layer_activations(
                layers,
                prepared_state
            ))

            probabilities = flatten(activation_collection[-1]["output"])
            action, _ = self.distribution.act(*probabilities)

            observation, reward, done, info = env.step(action)

            if done:
                state = env.reset()
                self.network.reset_states()
            else:
                state = observation

        self._data = tf.data.Dataset.from_tensor_slices({
            **stack_dicts(activation_collection),
            **stack_sensations(state_collection).dict(),
        })

        self.prepared = True

    def measure_predictability(self, source_layer: str, target_information: str):
        """Measure the predictability of target_information based on the information in source_layer's activation."""
        assert self.prepared, "Need to prepare before investigating."
        assert source_layer in self.list_layer_names(only_para_layers=False) + ["noise"]
        assert target_information in self._data.as_numpy_iterator().__next__().keys()

        output_dim = self._data.as_numpy_iterator().__next__()[target_information].shape[-1]

        predictor = tf.keras.Sequential([
            tf.keras.layers.Dense(100),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(100),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(output_dim),
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        predictor.compile(
            optimizer,
            loss="mse"
        )

        base_shape = list(self._data.as_numpy_iterator().__next__().values())[0].shape
        predictor.fit(
            self._data.map(lambda x:
                           (tf.random.normal(base_shape) if source_layer == "noise" else x[source_layer],
                            x[target_information])),
            batch_size=64,
            epochs=30,
            shuffle=True,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
            ]
        )
