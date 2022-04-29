from dexterity.analysis.investigators import Predictability

import tensorflow as tf

from dexterity.common.policies import BasePolicyDistribution
import tensorflow_probability as tfp


class Correlation(Predictability):

    def __init__(self, network: tf.keras.Model, distribution: BasePolicyDistribution):
        super().__init__(network, distribution)

    def measure(self, source_layer: str, target_information: str):
        """Measure the correlation of target_information with source_layer's activation."""
        assert self.prepared, "Need to prepare before investigating."
        assert source_layer in self.list_layer_names(only_para_layers=False) + ["noise"]
        assert target_information in self._data.as_numpy_iterator().__next__().keys()

        correlation = tfp.stats.correlation(
            x=self._data[source_layer],
            y=self._data[target_information],
            sample_axis=0
        )

        print(correlation)
