#!/usr/bin/env python
"""Tools to visualize the weights in any network."""
import itertools
import math
import os
from pprint import pprint
from typing import List

import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
from tqdm import tqdm

from utilities.util import normalize

CONVOLUTION_BASE_CLASS = tf.keras.layers.Conv2D.__bases__[0]


def plot_2d_filter_weights(filter_weights):
    # prepare subplots
    n_filters = filter_weights.shape[-1]
    tiles_per_row = math.ceil(math.sqrt(n_filters))
    fig, axes = plt.subplots(tiles_per_row, tiles_per_row)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    # normalize data
    filter_weights = normalize(filter_weights)

    i = 0
    for row_of_axes in axes:
        for axis in row_of_axes:
            if i < n_filters:
                axis.imshow(filter_weights[:, :, :, i])
            else:
                axis.axis("off")
            axis.set_xticks([])
            axis.set_yticks([])
            i += 1


def plot_image_tiling(images: List[numpy.ndarray]):
    # prepare subplots
    n_filters = len(images)
    tiles_per_row = math.ceil(math.sqrt(n_filters))
    fig, axes = plt.subplots(tiles_per_row, tiles_per_row)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    i = 0
    for row_of_axes in axes:
        for axis in row_of_axes:
            if i < n_filters:
                axis.imshow(images[i])
            else:
                axis.axis("off")
            axis.set_xticks([])
            axis.set_yticks([])
            i += 1


def extract_layers(network: tf.keras.Model) -> List[tf.keras.layers.Layer]:
    """Recursively extract layers from a potentially nested list of Sequentials of unknown depth."""
    return list(itertools.chain(*[extract_layers(layer)
                                  if isinstance(layer, tf.keras.Sequential)
                                  else [layer] for layer in network.layers]))


class WeightAnalyzer:

    def __init__(self, network: tf.keras.Model, mode: str = "show"):
        self.network = network
        self.mode = "show"
        self.set_mode(mode)

        self.figure_directory = "../docs/analysis/figures/"
        os.makedirs(self.figure_directory, exist_ok=True)

    def set_mode(self, mode: str):
        assert mode in ["show", "save"], "Illegal Analyzer Mode. Choose from show, save."
        self.mode = mode

    def list_layer_names(self) -> List[str]:
        return [layer.name for layer in extract_layers(self.network)]

    def list_convolutional_layer_names(self) -> List[str]:
        return [layer.name for layer in extract_layers(self.network) if isinstance(layer, CONVOLUTION_BASE_CLASS)]

    def list_non_convolutional_layer_names(self) -> List[str]:
        return [layer.name for layer in extract_layers(self.network) if
                not isinstance(layer, CONVOLUTION_BASE_CLASS) and not isinstance(layer, tf.keras.layers.Activation)]

    def plot_model(self):
        tf.keras.utils.plot_model(self.network, show_shapes=True)

    def visualize_layer_weights(self, layer_name):
        """Plot the weights (ignoring the bias) of a convolutional layer as a tessellation of its filters."""
        assert layer_name in self.list_convolutional_layer_names()
        layer = extract_layers(self.network)[self.list_layer_names().index(layer_name)]

        weights = normalize(layer.get_weights()[0])
        filters = [weights[:, :, :, f] for f in range(layer.output_shape[-1])]
        plot_image_tiling(filters)
        plt.show() if self.mode == "show" else plt.savefig(f"{self.figure_directory}/weights_{layer_name}.pdf",
                                                           format="pdf")

        plt.close()

    def visualize_max_filter_respondence(self, layer_name: str, feature_ids: List[int] = None):
        """Feature Maximization."""
        # build model that only goes to the layer of interest
        layer = extract_layers(self.network)[self.list_layer_names().index(layer_name)]
        print("performing feature maximization on " + (
            "dense " if not isinstance(layer, CONVOLUTION_BASE_CLASS) else "convolutional ") + "layer")
        intermediate_model = tf.keras.Model(inputs=self.network.input, outputs=layer.output)

        feature_ids = list(range(layer.output_shape[-1])) if feature_ids is None else feature_ids

        feature_maximizations = []
        for feature_id in tqdm(feature_ids, desc="Feature Maximization"):
            input_shape = (1,) + self.network.input_shape[1:]
            input_noise = tf.Variable(tf.random.uniform(input_shape, 0, 1), trainable=True)

            for _ in range(50):
                with tf.GradientTape() as tape:
                    activations = intermediate_model(input_noise)
                    if isinstance(layer, CONVOLUTION_BASE_CLASS):
                        loss = tf.reduce_sum(activations[:, :, :, feature_id])
                    else:
                        loss = activations[:, feature_id]

                gradient = tape.gradient(loss, input_noise)
                # step size from https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/13_Visual_Analysis.ipynb
                step_size = 100
                input_noise.assign_add(gradient * step_size)
                input_noise.assign(normalize(input_noise))

            final_image = tf.squeeze(input_noise).numpy()
            feature_maximizations.append(final_image)

        plot_image_tiling(feature_maximizations)
        plt.show() if self.mode == "show" else plt.savefig(
            f"{self.figure_directory}/feature_maximization_{layer_name}.pdf",
            format="pdf")

    def visualize_activation_map(self):
        pass

    # BUILDERS

    @staticmethod
    def from_saved_model(model_path: str, mode: str = "show"):
        assert os.path.exists(model_path), "Model Path does not exist!"

        return WeightAnalyzer(tf.keras.models.load_model(model_path), mode=mode)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    tf.random.set_seed(1)

    model = tf.keras.models.load_model(
        "../saved_models/pretrained_components/visual_component/classification/pretrained_encoder.h5")
    analyzer = WeightAnalyzer(model, mode="save")
    pprint(analyzer.list_layer_names())

    analyzer.visualize_max_filter_respondence("dense_1")
