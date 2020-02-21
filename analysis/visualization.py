#!/usr/bin/env python
"""Tools to visualize the weights in any network."""
import math
import os
from typing import List, Iterable, Any, Union

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
from sklearn.manifold import TSNE
from tensorflow_core.python.keras.utils import plot_model
from tqdm import tqdm
import tensorflow_datasets as tfds

from utilities.util import normalize
from utilities.model_utils import extract_layers

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


def is_conv(layer):
    """Check if layer is convolutional."""
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
        """ Set the mode of visualizations to either "show" or "save".

        Args:
            mode: either one of [show, save], where show creates popups for results and save saves them into the figure
                directory
        """
        assert mode in ["show", "save"], "Illegal Analyzer Mode. Choose from show, save."
        self.mode = mode

    def numb_features_in_layer(self, layer_name: str) -> int:
        """Get the number of features/filters in the layer specified by its unique string representation."""
        return self.get_layer_by_name(layer_name).output_shape[-1]

    def list_layer_names(self, only_para_layers=False) -> List[str]:
        """Get a list of unique string representations of all layers in the network."""
        if only_para_layers:
            return [layer.name for layer in extract_layers(self.network) if not isinstance(layer, tf.keras.layers.Activation)]
        else:
            return [layer.name for layer in extract_layers(self.network)]

    def list_convolutional_layer_names(self) -> List[str]:
        """Get a list of unique string representations of convolutional layers in the network."""
        return [layer.name for layer in extract_layers(self.network) if is_conv(layer)]

    def list_non_convolutional_layer_names(self) -> List[str]:
        """Get a list of unique string representations of non-convolutional layers in the network."""
        return [layer.name for layer in extract_layers(self.network) if
                not isinstance(layer, CONVOLUTION_BASE_CLASS) and not isinstance(layer, tf.keras.layers.Activation)]

    def get_layer_by_name(self, layer_name):
        """Retrieve the layer object identified from the model by its unique string representation."""
        return extract_layers(self.network)[self.list_layer_names().index(layer_name)]

    def plot_model(self):
        """Plot the network graph into a file."""
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
                                         optimization_steps: int = 30) -> None:
        """Visualize a given layer by performing feature maximization.

        In feature maximization, a random input image is created (here, uniform noise) and then successively updated
        to maximize the response in a filter (feature) of the specified layer(s). The procedure is extended to start
        with a low-resolution input image that is upscaled after every 30 optimization steps by some factor. This
        encourages the optimization to go towards a local minimum with low-frequency patterns, which are generally
        easier to interpret.

        Args:
            layer_name (str): the unique string identifying the layer who's response shall be maximized
            feature_ids (list): a list of filter/feature IDs that will be maximized and shown as a tiling
            optimization_steps: the number of optimization steps between upscalings.

        Returns:
            None
        """
        # build model that only goes to the layer of interest
        layer = self.get_layer_by_name(layer_name)
        print("performing feature maximization on " + ("dense " if not isinstance(layer, CONVOLUTION_BASE_CLASS)
                                                       else "convolutional ") + "layer " + layer_name + ".")
        intermediate_model = tf.keras.Model(inputs=self.network.input, outputs=layer.output)

        feature_ids = list(range(layer.output_shape[-1])) if feature_ids is None or feature_ids == [] else feature_ids
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

        plt.clf()
        plot_image_tiling(feature_maximizations)
        plt.show() if self.mode == "show" else plt.savefig(
            f"{self.figure_directory}/feature_maximization_{layer_name}_{'_'.join(map(str, feature_ids))}.pdf",
            format="pdf")

    def visualize_activation_map(self, layer_name: str, reference_img: numpy.ndarray, mode: str = "gray") -> None:
        """Visualize the activation map of a given layer, either as a gray level image, as a heatmap on top of the
        original image, or as a bar plot.

        Args:
            layer_name (str): the unique string identifying the layer from which to draw the activations
            reference_img: the image serving as the input producing the activation
            mode: the mode of the visualization, either one of [gray, heat, plot]

        Returns:
            None
        """
        assert mode in ["gray", "heat", "plot"]

        layer = self.get_layer_by_name(layer_name)
        sub_model = tf.keras.Model(inputs=self.network.input, outputs=layer.output)

        reference_img = normalize(reference_img).astype(numpy.float32)
        reference_img = tf.expand_dims(reference_img, axis=0) if len(reference_img.shape) == 3 else reference_img
        reference_width, reference_height = reference_img.shape[1], reference_img.shape[2]

        feature_maps = sub_model(reference_img)
        n_filters = feature_maps.shape[-1]

        plt.clf()
        if mode == "heat":
            feature_maps = tf.image.resize(feature_maps, size=(reference_width, reference_height), method="bicubic")
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
            feature_maps = feature_maps.numpy()
            plot_image_tiling([numpy.squeeze(feature_maps[:, :, :, i]) for i in range(layer.output_shape[-1])],
                              cmap="gray")
        elif mode == "plot":
            feature_maps = tf.squeeze(feature_maps)
            mean_filter_responses = tf.reduce_mean(feature_maps, axis=[0, 1])
            plt.bar(list(range(n_filters)), mean_filter_responses)

        output_format = "png" if mode in ["heat", "gray"] else "pdf"
        plt.show() if self.mode == "show" else plt.savefig(
            f"{self.figure_directory}/feature_maps_{layer_name}{f'_{mode}'}.{output_format}",
            format=output_format, dpi=300)

        return feature_maps

    def obtain_saliency_map(self, reference: Union[numpy.ndarray, tf.Tensor]) -> None:
        """Create a saliency map indicating the importance of each input pixel for the output, based on gradients.

        Args:
            reference: the image serving as a reference image
        """

        # resize image to fit network input shape
        reference = tf.Variable(tf.image.resize(reference, size=self.network.input_shape[1:-1]))

        # obtain output layer and change its activation to linear
        last_layer = self.get_layer_by_name(self.list_layer_names(only_para_layers=True)[-1])
        last_layer_input_shape = (1,) + last_layer.input_shape[1:]
        last_layer_config = tf.keras.layers.serialize(last_layer)
        last_layer_config["config"]["activation"] = "linear"
        new_last_layer = tf.keras.layers.deserialize(last_layer_config)
        new_last_layer(tf.random.normal(last_layer_input_shape))
        new_last_layer.set_weights(last_layer.get_weights())

        # create a new model with the linear output
        prev_last_layer = self.get_layer_by_name(self.list_layer_names(only_para_layers=True)[-2])
        new_last_layer_output = new_last_layer(prev_last_layer.output)
        submodel = tf.keras.Model(inputs=self.network.input, outputs=new_last_layer_output)

        # make prediction based on new layer
        with tf.GradientTape() as tape:
            linear_prediction = submodel(tf.dtypes.cast(tf.expand_dims(reference, axis=0), dtype=tf.float32))
            max_feature = tf.argmax(linear_prediction, axis=1)
            loss = -linear_prediction[:, int(max_feature.numpy().item())]

        output_gradient = tape.gradient(loss, reference)

        saliency_map = tf.reduce_mean(tf.keras.activations.relu(output_gradient), axis=-1)

        plt.clf()
        plt.imshow(saliency_map, cmap="jet")
        plt.show() if self.mode == "show" else plt.savefig(
            f"{self.figure_directory}/saliency_map.pdf", format="pdf", dpi=300)

    def cluster_inputs(self, layer_name: str, input_images: List[numpy.ndarray], classes: Iterable[int]) -> None:
        """Cluster inputs based on the representation produced by a given layer, using the t-SNE algorithm.

        Args:
            layer_name (str): the unique string identifying the layer from which to take the representation.
            input_images (list): a list of images that will be clustered
            classes (list): the classes/categories of the images based on which they are colored in the cluster plot
        """
        layer = self.get_layer_by_name(layer_name)
        sub_model = tf.keras.Model(inputs=self.network.input, outputs=layer.output)

        # retrieve representations
        input_images = [tf.expand_dims(normalize(img).astype(numpy.float32), axis=0) for img in input_images]
        representations = [tf.reshape(sub_model(img), [-1]).numpy() for img in input_images]

        # cluster
        t_sne = TSNE()
        sneezed_datapoints = t_sne.fit_transform(representations)
        x, y = numpy.split(sneezed_datapoints, 2, axis=1)

        plt.clf()
        plt.scatter(x=numpy.squeeze(x), y=numpy.squeeze(y), c=classes, cmap="Paired")

        plt.show() if self.mode == "show" else plt.savefig(
            f"{self.figure_directory}/input_cluster_{layer_name}.pdf", format="pdf", dpi=300)

    @staticmethod
    def from_saved_model(model_path: str, mode: str = "show"):
        """Build the analyzer from a model path."""
        assert os.path.exists(model_path), "Model Path does not exist!"
        return NetworkAnalyzer(tf.keras.models.load_model(model_path), mode=mode)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    f_weights = False
    f_max = False
    f_map = True
    f_salience = True
    do_tsne = False

    tf.random.set_seed(1)

    model = tf.keras.applications.VGG16()
    analyzer = NetworkAnalyzer(model, mode="save")
    print(analyzer.list_layer_names())

    if f_weights:
        analyzer.visualize_layer_weights("block1_conv1")

    # FEATURE MAXIMIZATION
    if f_max:
        n_features = analyzer.numb_features_in_layer("block2_conv2")
        analyzer.visualize_max_filter_respondence("fc1", feature_ids=[])

    # FEATURE MAPS
    if f_map:
        reference = mpimg.imread("hand.png")
        analyzer.visualize_activation_map("block1_conv1", reference, mode="gray")

    # SALIENCE
    if f_salience:
        reference = mpimg.imread("hand.png")
        analyzer.obtain_saliency_map(reference)

    # T-SNE CLUSTERING
    if do_tsne:
        data, _ = tfds.load("cifar10", shuffle_files=True).values()
        data = data.map(lambda img: (tf.image.resize(img["image"], (224, 224)) / 255, img["label"]))
        data = list(iter(data.take(100)))
        images = [d[0] for d in data]
        classes = [d[1].numpy().item() for d in data]

        analyzer.cluster_inputs("predictions", images, classes)

