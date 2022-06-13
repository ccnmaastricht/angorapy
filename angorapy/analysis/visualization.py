#!/usr/bin/env python
"""Tools to visualize the weights/filters/neurons in a (convolutional) network."""
import math
import os
from typing import List, Iterable, Union

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from sklearn.manifold import TSNE
from tqdm import tqdm

from analysis.investigation import Investigator
from angorapy.pretrain import top_5_accuracy
from utilities.model_utils import extract_layers, build_sub_model_to, is_conv
from utilities.plotting import plot_image_tiling, transparent_cmap
from utilities.util import normalize


class Visualizer(Investigator):
    """Utilities for visualizing network layers/filters."""

    def __init__(self, network: tf.keras.Model, mode: str = "show"):
        """

        Args:
            network: the netowrk to be visualized
            mode: either one of [show, save, return], where show creates popups for results and save saves them into the
                figure directory, return does neither and only returns result
        """
        super().__init__(network, None)
        self.network = network
        self.mode = "show"
        self.set_mode(mode)

        self.figure_directory = "../docs/analysis/figures/"
        os.makedirs(self.figure_directory, exist_ok=True)

    def set_mode(self, mode: str):
        """ Set the mode of visualizations to either "show" or "save".

        Args:
            mode: either one of [show, save, return], where show creates popups for results and save saves them into the
                figure directory, return does neither and only returns result
        """
        assert mode in ["show", "save", "return"], "Illegal Analyzer Mode. Choose from show, save."
        self.mode = mode

    def numb_features_in_layer(self, layer_name: str) -> int:
        """Get the number of features/filters in the layer specified by its unique string representation."""
        return self.get_layer_by_name(layer_name).output_shape[-1]

    def list_layer_names(self, only_para_layers=False) -> List[str]:
        """Get a list of unique string representations of all layers in the network."""
        if only_para_layers:
            return [layer.name for layer in extract_layers(self.network) if
                    not isinstance(layer, tf.keras.layers.Activation)]
        else:
            return [layer.name for layer in extract_layers(self.network)]

    def layer_weights_plot(self, layer_name):
        """Plot the weights (ignoring the bias) of a convolutional layer as a tessellation of its filters."""
        assert layer_name in self.list_convolutional_layer_names()
        layer = extract_layers(self.network)[self.list_layer_names().index(layer_name)]

        weights = normalize(layer.get_weights()[0])
        filters = [weights[:, :, :, f] for f in range(layer.output_shape[-1])]
        plot_image_tiling(filters)
        plt.show() if self.mode == "show" else plt.savefig(f"{self.figure_directory}/weights_{layer_name}.pdf",
                                                           format="pdf")

        plt.close()

    def preferred_stimulus(self, layer_name: str, feature_ids: List[int] = None,
                           optimization_steps: int = 500) -> List:
        """Visualize a given layer by performing feature maximization.

        Explanation at: http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture12.pdf

        In feature maximization, a random input image is created (here, uniform noise) and then successively updated
        to maximize the response in a filter (feature) of the specified layer(serialization). The procedure is extended to start
        with a low-resolution input image that is upscaled after every 30 optimization steps by some factor. This
        encourages the optimization to go towards a local minimum with low-frequency patterns, which are generally
        easier to interpret.

        Args:
            layer_name (str): the unique string identifying the layer who'serialization response shall be maximized
            feature_ids (list): a list of filter/feature IDs that will be maximized and shown as a tiling
            optimization_steps: the number of optimization steps between upscalings.

        Returns:
            None
        """
        # build model that only goes to the layer of interest

        # obtain output layer and change its activation to linear
        layer = self.get_layer_by_name(layer_name)

        if "activation" in layer.get_config().keys() and layer.get_config()["activation"] == "softmax":
            layer_input_shape = (1,) + layer.input_shape[1:]
            layer_config = tf.keras.layers.serialize(layer)
            layer_config["config"]["activation"] = "linear"
            new_layer = tf.keras.layers.deserialize(layer_config)
            new_layer(tf.random.normal(layer_input_shape))
            new_layer.set_weights(layer.get_weights())

            # create a new model with the linear output
            prev_layer = self.get_layer_by_name(self.list_layer_names(only_para_layers=True)[-2])
            new_layer_output = new_layer(prev_layer.output)
            intermediate_model = tf.keras.Model(inputs=self.network.input, outputs=new_layer_output)
        else:
            intermediate_model = tf.keras.Model(inputs=self.network.input, outputs=layer.output) # build_sub_model_to(self.network, [layer])

        # intermediate_model = tf.keras.Model(inputs=self.network.input, outputs=layer.output)
        width_in, height_in = intermediate_model.input_shape[1], intermediate_model.input_shape[2]

        feature_ids = list(range(layer.output_shape[-1])) if feature_ids is None or feature_ids == [] else feature_ids
        feature_maximizations = []
        for feature_id in tqdm(feature_ids, desc="Feature Maximization", disable=True):
            image = tf.zeros((1, width_in, height_in, 3)) + 0.5 + tf.random.normal((1, width_in, height_in, 3), 0, 0.03)

            optimizer = tf.keras.optimizers.Adam(0.1)
            image_variable = tf.Variable(image, trainable=True)

            for i in tqdm(range(optimization_steps), desc=f"{layer_name} {feature_id}"):

                with tf.GradientTape() as tape:
                    activations = intermediate_model(image_variable)
                    if isinstance(activations, list):
                        activations = activations[0]
                    loss = - tf.reduce_mean(activations[:, :, :, feature_id]) if is_conv(layer) else activations[:,
                                                                                                     feature_id]
                    # add l2 norm regularization
                    loss += 0.0001 * tf.reduce_sum(tf.square(image))

                gradient = tape.gradient(loss, image_variable)
                optimizer.apply_gradients([(gradient, image_variable)])

                image_tensor = tf.convert_to_tensor(image_variable)
                if i < optimization_steps - 1 and i % 4 == 0:
                    image_tensor = tfa.image.mean_filter2d(image_tensor, 3, padding="REFLECT")

                image_variable = tf.Variable(image_tensor, trainable=True)

            image = tf.convert_to_tensor(image_variable)
            final_image = normalize(tf.squeeze(image).numpy())
            feature_maximizations.append(final_image)

        plt.clf()
        plot_image_tiling(feature_maximizations)

        if self.mode == "show":
            plt.show()
        elif self.mode == "save":
            plt.savefig(
                f"{self.figure_directory}/feature_maximization_{layer_name}_{'_'.join(map(str, feature_ids))}.pdf",
                format="pdf")

        return feature_maximizations

    def feature_map(self, layer_name: str, reference_img: numpy.ndarray, mode: str = "gray"):
        """Visualize the activation map of a given layer, either as a gray level image, as a heatmap on top of the
        original image, or as a bar plot.

        Args:
            layer_name (str): the unique string identifying the layer from which to draw the activations
            reference_img: the image serving as the input producing the activation
            mode: the mode of the visualization, either one of [gray, heat, plot]

        Returns:
            None
        """
        assert mode in ["gray", "grey", "heat", "plot"]

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

        elif mode in ["gray", "grey"]:
            feature_maps = feature_maps.numpy()
            fig, axes = plot_image_tiling(
                [numpy.squeeze(feature_maps[:, :, :, i]) for i in range(layer.output_shape[-1])],
                cmap="gray")
        elif mode == "plot":
            feature_maps = tf.squeeze(feature_maps)
            mean_filter_responses = tf.reduce_mean(feature_maps, axis=[0, 1])
            fig, axes = plt.subplots()
            axes.bar(list(range(n_filters)), mean_filter_responses)

        else:
            raise NotImplementedError("WhAT iS thIS?")

        output_format = "png" if mode in ["heat", "gray", "grey"] else "pdf"

        if self.mode == "show":
            plt.show()
        elif self.mode == "save":
            plt.savefig(f"{self.figure_directory}/feature_maps_{layer_name}{f'_{mode}'}.{output_format}",
                        format=output_format, dpi=300)

        plt.clf()
        plt.cla()

        return feature_maps

    def saliency_map(self, reference: Union[numpy.ndarray, tf.Tensor], layer_name=None, neuron: int = None) -> None:
        """Create a saliency map indicating the importance of each input pixel for the output, based on gradients.
        Only works for fully connected layers!

        Explanation at: http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture12.pdf

        Args:
            reference: the image serving as a reference image
            layer_name: string identifier of the layer in which a neuron will be analyzed, if None, use last layer
            neuron: neuron index in the layer, if None use the one with the highest activity
        """

        # resize image to fit network input shape
        reference = tf.Variable(tf.image.resize(reference, size=self.network.input_shape[1:-1]) / 255)

        if layer_name is None:
            layer = self.get_layer_by_name(self.list_layer_names(only_para_layers=True)[-1])
            layer_index = -1
        else:
            layer = self.get_layer_by_name(layer_name)
            layer_index = self.list_layer_names(only_para_layers=True).index(layer_name)

        if is_conv(layer):
            raise ValueError("Cannot do saliency maps for convolutional layers!")

        if "activation" in layer.get_config().keys() and layer.get_config()["activation"] == "softmax":
            layer_input_shape = (1,) + layer.input_shape[1:]
            layer_config = tf.keras.layers.serialize(layer)
            layer_config["config"]["activation"] = "linear"
            new_layer = tf.keras.layers.deserialize(layer_config)
            new_layer(tf.random.normal(layer_input_shape))
            new_layer.set_weights(layer.get_weights())

            # create a new model with the linear output
            prev_layer = self.get_layer_by_name(self.list_layer_names(only_para_layers=True)[layer_index - 1])
            new_layer_output = new_layer(prev_layer.output)
            intermediate_model = tf.keras.Model(inputs=self.network.input, outputs=new_layer_output)
        else:
            intermediate_model = build_sub_model_to(self.network, [layer])

        # make prediction based on new layer
        with tf.GradientTape() as tape:
            linear_prediction = intermediate_model(tf.dtypes.cast(tf.expand_dims(reference, axis=0), dtype=tf.float32))
            if isinstance(linear_prediction, list):
                linear_prediction = linear_prediction[0]

            if neuron is None:
                max_feature = int(tf.argmax(linear_prediction, axis=1).numpy().item())
            else:
                max_feature = neuron
            loss = tf.sigmoid(linear_prediction[:, max_feature])

        output_gradient = tape.gradient(loss, reference)
        saliency_map = tf.reduce_max(tf.math.abs(output_gradient), axis=-1)

        if self.mode != "return":
            plt.imshow(saliency_map, cmap="jet")

        if self.mode == "show":
            plt.show()
        elif self.mode == "save":
            plt.savefig(f"{self.figure_directory}/saliency_map.pdf", format="pdf", dpi=300)

        return saliency_map

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
        return Visualizer(tf.keras.models.load_model(model_path), mode=mode)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    f_weights = False
    f_max = False
    f_map = False
    f_salience = True
    do_tsne = False

    tf.random.set_seed(69420)

    model = tf.keras.models.load_model("../storage/pretrained/visual_r.h5",
                                       custom_objects={"top_5_accuracy": top_5_accuracy})
    analyzer = Visualizer(model, mode="show")

    # WEIGHT VISUALIZATION
    if f_weights:
        analyzer.layer_weights_plot("conv2d")

    # FEATURE MAXIMIZATION
    if f_max:
        fids = [5]
        analyzer.preferred_stimulus("conv2d_4", feature_ids=fids)

    # FEATURE MAPS
    if f_map:
        layer = "conv2d_3"
        reference = mpimg.imread("img/hand.png")
        analyzer.feature_map(layer, reference, mode="gray")
        analyzer.feature_map(layer, reference, mode="heat")
        analyzer.feature_map(layer, reference, mode="plot")

    # SALIENCE
    if f_salience:
        reference = mpimg.imread("img/hand.png")
        for i in range(10):
            analyzer.saliency_map(reference, layer_name="dense_1", neuron=i)

    # T-SNE CLUSTERING
    if do_tsne:
        data, _ = tfds.load("cifar10", shuffle_files=True).values()
        data = data.map(lambda img: (tf.image.resize(img["image"], (224, 224)) / 255, img["label"]))
        data = list(iter(data.take(100)))
        images = [d[0] for d in data]
        classes = [d[1].numpy().item() for d in data]

        analyzer.cluster_inputs("predictions", images, classes)
