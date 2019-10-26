#!/usr/bin/env python
"""Tools to visualize the weights in any network."""
import itertools
import math
import os
from pprint import pprint
from typing import List

import matplotlib.pyplot as plt
import tensorflow as tf

from models.components import VisualComponent


CONVOLUTION_BASE_CLASS = tf.keras.layers.Conv2D.__bases__[0]


def plot_2d_filter_weights(filter_weights):
    # prepare subplots
    n_filters = filter_weights.shape[-1]
    tiles_per_row = math.ceil(math.sqrt(n_filters))
    fig, axes = plt.subplots(tiles_per_row, tiles_per_row)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    # normalize data
    filter_weights = (filter_weights - filter_weights.min()) / (filter_weights.max() - filter_weights.min())

    i = 0
    for row_of_axes in axes:
        for axis in row_of_axes:
            if i < n_filters:
                axis.imshow(filter_weights[:, :, :, i])
            axis.axis("off")
            i += 1


def extract_layers(network: tf.keras.Model) -> List[tf.keras.layers.Layer]:
    """Recursively extract layers from a potentially nested list of Sequentials of unknown depth."""
    return list(itertools.chain(*[extract_layers(layer)
                                  if isinstance(layer, tf.keras.Sequential)
                                  else [layer] for layer in network.layers]))


class WeightAnalyzer:

    def __init__(self, network: tf.keras.Model):
        self.network = network

    def list_layer_names(self) -> List[str]:
        return [layer.name for layer in extract_layers(self.network)]

    def list_convolutional_layer_names(self) -> List[str]:
        return [layer.name for layer in extract_layers(self.network) if isinstance(layer, CONVOLUTION_BASE_CLASS)]

    def list_non_convolutional_layer_names(self) -> List[str]:
        return [layer.name for layer in extract_layers(self.network) if not isinstance(layer, CONVOLUTION_BASE_CLASS)]

    def visualize_layer_weights(self, layer_name):
        assert layer_name in self.list_convolutional_layer_names()
        layer = extract_layers(self.network)[self.list_layer_names().index(layer_name)]

        plot_2d_filter_weights(layer.get_weights()[0])
        plt.show()


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    model = VisualComponent()
    model.predict(tf.random.normal([1, 200, 200, 3]))
    analyzer = WeightAnalyzer(model)

    pprint(analyzer.list_convolutional_layer_names())
    analyzer.visualize_layer_weights("conv2d")