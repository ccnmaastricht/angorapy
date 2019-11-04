#!/usr/bin/env python
"""Tools to visualize the weights in any network."""
import itertools
import math
import os
from typing import List

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
from tqdm import tqdm

from utilities.util import normalize

CONVOLUTION_BASE_CLASS = tf.keras.layers.Conv2D.__bases__[0]


def plot_image_tiling(images: List[numpy.ndarray], cmap: str = None):
    # prepare subplots
    n_filters = len(images)
    tiles_per_row = math.ceil(math.sqrt(n_filters))
    fig, axes = plt.subplots(tiles_per_row, tiles_per_row)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    i = 0
    axes = [[axes]] if not isinstance(axes, numpy.ndarray) else axes
    for row_of_axes in axes:
        for axis in row_of_axes:
            if i < n_filters:
                axis.imshow(images[i], cmap=cmap) if cmap is not None else axis.imshow(images[i])
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


def is_conv(layer):
    return isinstance(layer, CONVOLUTION_BASE_CLASS)


def transparent_cmap(cmap, N=255):
    """https://stackoverflow.com/questions/42481203/heatmap-on-top-of-image"""
    cmap._init()
    cmap._lut[:, -1] = numpy.linspace(0, 0.8, N + 4)
    return cmap


class NetworkAnalyzer:

    def __init__(self, network: tf.keras.Model, mode: str = "show"):
        self.network = network
        self.mode = "show"
        self.set_mode(mode)

        self.figure_directory = "../docs/analysis/figures/"
        os.makedirs(self.figure_directory, exist_ok=True)

    def set_mode(self, mode: str):
        assert mode in ["show", "save"], "Illegal Analyzer Mode. Choose from show, save."
        self.mode = mode

    def numb_features_in_layer(self, layer: str):
        return self.get_layer_by_name(layer).output_shape[-1]

    def list_layer_names(self) -> List[str]:
        return [layer.name for layer in extract_layers(self.network)]

    def list_convolutional_layer_names(self) -> List[str]:
        return [layer.name for layer in extract_layers(self.network) if is_conv(layer)]

    def list_non_convolutional_layer_names(self) -> List[str]:
        return [layer.name for layer in extract_layers(self.network) if
                not isinstance(layer, CONVOLUTION_BASE_CLASS) and not isinstance(layer, tf.keras.layers.Activation)]

    def get_layer_by_name(self, layer_name):
        return extract_layers(self.network)[self.list_layer_names().index(layer_name)]

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

    def visualize_max_filter_respondence(self, layer_name: str, feature_ids: List[int] = None,
                                         optimization_steps: int = 30):
        """Feature Maximization.
        https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030"""
        # build model that only goes to the layer of interest
        input_shape = (1,) + self.network.input_shape[1:]
        layer = self.get_layer_by_name(layer_name)
        print("performing feature maximization on " + ("dense " if not isinstance(layer, CONVOLUTION_BASE_CLASS)
                                                       else "convolutional ") + "layer " + layer_name + ".")
        intermediate_model = tf.keras.Model(inputs=self.network.input, outputs=layer.output)

        feature_ids = list(range(layer.output_shape[-1])) if feature_ids is None else feature_ids
        feature_maximizations = []
        for feature_id in tqdm(feature_ids, desc="Feature Maximization"):
            current_size = 56
            image = tf.random.uniform((1, current_size, current_size, 3), 0, 1)

            for _ in range(12):
                optimizer = tf.keras.optimizers.Adam(0.1)
                image_variable = tf.Variable(image, trainable=True)

                for _ in range(optimization_steps):
                    with tf.GradientTape() as tape:
                        activations = intermediate_model(image_variable)
                        loss = tf.reduce_mean(activations[:, :, :, feature_id]) if is_conv(layer) else activations[:,
                                                                                                       feature_id]
                        loss += 1e-6 * tf.reduce_sum(tf.square(image))

                    gradient = tape.gradient(loss, image_variable)
                    optimizer.apply_gradients([(gradient, image_variable)])

                current_size = int(current_size * 1.2)
                image = tf.convert_to_tensor(image_variable)
                image = tf.image.resize(image, (current_size, current_size), method="bicubic")

            final_image = tf.clip_by_value(tf.squeeze(image), 0, 1).numpy()
            feature_maximizations.append(final_image)

        plot_image_tiling(feature_maximizations)
        plt.mlpd3() if self.mode == "show" else plt.savefig(
            f"{self.figure_directory}/feature_maximization_{layer_name}_{'_'.join(map(str, feature_ids))}.pdf",
            format="pdf")

    def visualize_activation_map(self, layer_name, reference_img: numpy.ndarray, mode: str = "gray"):
        assert mode in ["gray", "heat", "plot"]

        layer = self.get_layer_by_name(layer_name)
        submodel = tf.keras.Model(inputs=self.network.input, outputs=layer.output)

        reference_img = normalize(reference_img).astype(numpy.float32)
        reference_img = tf.expand_dims(reference_img, axis=0) if len(reference_img.shape) == 3 else reference_img
        reference_width, reference_height = reference_img.shape[1], reference_img.shape[2]

        feature_maps = submodel(reference_img)
        n_filters = feature_maps.shape[-1]

        if mode == "heat":
            feature_maps = tf.image.resize(feature_maps, size=(reference_width, reference_height))
            feature_maps = tf.squeeze(feature_maps)

            # prepare subplots
            tiles_per_row = math.ceil(math.sqrt(n_filters))
            fig, axes = plt.subplots(tiles_per_row, tiles_per_row)
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            hm_cmap = transparent_cmap(plt.cm.get_cmap("hot"))

            i = 0
            for row_of_axes in axes:
                for axis in row_of_axes:
                    if i < n_filters:
                        axis.imshow(tf.squeeze(tf.image.rgb_to_grayscale(reference_img)), cmap="gray")
                        axis.contourf(feature_maps[:, :, i], cmap=hm_cmap)
                    else:
                        axis.axis("off")
                    axis.set_xticks([])
                    axis.set_yticks([])
                    i += 1
        elif mode == "gray":
            plot_image_tiling([tf.squeeze(feature_maps[:, :, :, i]) for i in range(layer.output_shape[-1])],
                              cmap="gray")
        elif mode == "plot":
            feature_maps = tf.squeeze(feature_maps)
            mean_filter_responses = tf.reduce_mean(feature_maps, axis=[0, 1])
            plt.bar(list(range(n_filters)), mean_filter_responses)

        format = "png" if mode == "heat" else "pdf"
        plt.show() if self.mode == "show" else plt.savefig(
            f"{self.figure_directory}/feature_maps_{layer_name}{f'_{mode}'}.{format}",
            format=format, dpi=300)

        return feature_maps

    # BUILDERS

    @staticmethod
    def from_saved_model(model_path: str, mode: str = "show"):
        assert os.path.exists(model_path), "Model Path does not exist!"

        return NetworkAnalyzer(tf.keras.models.load_model(model_path), mode=mode)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    f_weights = False
    f_max = False
    f_map = True

    tf.random.set_seed(1)

    model = tf.keras.applications.VGG16()
    analyzer = NetworkAnalyzer(model, mode="save")
    print(analyzer.list_layer_names())

    if f_weights:
        analyzer.visualize_layer_weights("block1_conv1")

    # FEATURE MAXIMIZATION
    if f_max:
        n_features = analyzer.numb_features_in_layer("block2_conv2")
        analyzer.visualize_max_filter_respondence("block2_conv2", feature_ids=[24, 57, 89, 120, 42, 26, 45, 21, 99])

    # FEATURE MAPS
    if f_map:
        reference = mpimg.imread("hase.jpg")
        analyzer.visualize_activation_map("block1_conv2", reference, mode="gray")
