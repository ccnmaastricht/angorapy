from angorapy.analysis.investigators import Predictability

import tensorflow as tf

from angorapy.common.policies import BasePolicyDistribution
import tensorflow_probability as tfp


class Correlation(Predictability):

    def __init__(self, network: tf.keras.Model, distribution: BasePolicyDistribution):
        super().__init__(network, distribution)

    def fit(self, source_layer: str, target_information: str):
        """Measure the correlation of target_information with source_layer's activation."""
        assert self.prepared, "Need to prepare before investigating."
        assert source_layer in self.list_layer_names(only_para_layers=False) + ["noise"]
        assert target_information in self._data.as_numpy_iterator().__next__().keys()

        all_cross_correlations = []
        for source, target in zip(list(self._data.map(lambda x: x[source_layer]).as_numpy_iterator()),
                                  list(self._data.map(lambda x: x[target_information]).as_numpy_iterator())):

            source = source[..., tf.newaxis]
            target = target[..., tf.newaxis, tf.newaxis]

            cross_correlation = tf.reduce_mean(tf.nn.convolution(source, target))
            all_cross_correlations.append(cross_correlation)

        mean_correlation = tf.reduce_mean(all_cross_correlations)

        print(f"{source_layer} -> {target_information}: {mean_correlation}")
