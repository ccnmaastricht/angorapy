from inspect import getfullargspec as fargs
from typing import List, Union, Callable

import numpy
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed

import angorapy
from angorapy.common.policies import BasePolicyDistribution
from angorapy.common.wrappers import BaseWrapper
from angorapy.utilities.error import IncompatibleModelException
from angorapy.utilities.util import flatten


def is_recurrent_model(model: tf.keras.Model):
    """Check if given model is recurrent (i.e. contains a recurrent layer of any sort)"""
    for layer in model.submodules:
        if isinstance(layer, tf.keras.layers.RNN):
            return True

    return False


def requires_batch_size(model_builder) -> bool:
    """Check if model building function requires a batch size when building."""
    return "bs" in fargs(model_builder).args + fargs(model_builder).kwonlyargs


def requires_sequence_length(model_builder) -> bool:
    """Check if model building function requires a sequenc_length when building."""
    return "sequence_length" in fargs(model_builder).args + fargs(model_builder).kwonlyargs


def list_layer_names(network, only_para_layers=True) -> List[str]:
    """Get a list of unique string representations of all layers in the network."""
    if only_para_layers:
        return [layer.name for layer in network.submodules if
                not isinstance(layer, (tf.keras.layers.Activation,
                                       tf.keras.layers.InputLayer,
                                       tf.keras.layers.Concatenate,
                                       tf.keras.layers.Reshape))]
    else:
        return [layer.name for layer in network.submodules if
                not isinstance(layer, tf.keras.layers.InputLayer)]


def get_layer_names(model: tf.keras.Model):
    """Get names of all (outer) layers in the model."""
    return [layer.name if not isinstance(layer, TimeDistributed) else layer.layer.name for layer in model.layers]


def extract_layers(network: tf.keras.Model, unfold_tds: bool = False) -> List[tf.keras.layers.Layer]:
    """Recursively extract layers from a potentially nested list of Sequentials of unknown depth."""
    if not hasattr(network, "layers"):
        return [network]

    layers = []
    for l in network.submodules:
        if isinstance(l, tf.keras.Model) or isinstance(l, tf.keras.Sequential):
            layers.append(extract_layers(l))
        elif isinstance(l, tf.keras.layers.TimeDistributed) and unfold_tds:
            if isinstance(l.layer, tf.keras.Model) or isinstance(l.layer, tf.keras.Sequential):
                layers.append(extract_layers(l.layer))
            else:
                layers.append(l.layer)
        else:
            layers.append(l)

    return flatten(layers)


def get_layers_by_names(network: tf.keras.Model, layer_names: List[str]):
    """Get a list of layers identified by their names from a network."""
    layers = network.submodules
    all_layer_names = [l.name for l in layers]

    assert all(ln in all_layer_names for ln in layer_names), \
        f"Cannot find layers {list(filter(lambda ln: ln not in all_layer_names, layer_names))} in network extraction."

    return [layers[all_layer_names.index(layer_name)] for layer_name in layer_names]


def get_parent_layer(child, network):
    if child in network.layers:
        return network

    for layer in network.submodules:
        if hasattr(layer, "layers") and child in layer.layers:
            return layer

    return None


def build_sub_model_to(network: tf.keras.Model, tos: Union[List[str], List[tf.keras.Model]], include_original=False):
    """Build a sub model of a given network that has (multiple) outputs at layer activations defined by a list of layer
    names."""
    layers = get_layers_by_names(network, tos) if isinstance(tos[0], str) else tos
    layer = layers[0]

    if not hasattr(layer, "layers") and layer in network.layers:
        return tf.keras.Model(inputs=network.inputs, outputs=layer.output)

    if hasattr(layer, "layers") and layer in network.layers:
        layer = layer.layers[-1]

    parent = get_parent_layer(layer, network)
    inbound_layers = parent.inbound_nodes[0].inbound_layers
    if not isinstance(inbound_layers, list):
        inbound_layers = [inbound_layers]

    outputs = [il.output if not hasattr(il, "layers") else build_sub_model_to(network, [il])(network.inputs) for il in inbound_layers]

    subnetwork_to_parent = tf.keras.Model(inputs=network.inputs, outputs=outputs)
    subnetwork_to_layer = tf.keras.Model(inputs=parent.inputs, outputs=layer.output)

    return tf.keras.Model(inputs=subnetwork_to_parent.inputs, outputs=subnetwork_to_layer(subnetwork_to_parent.output))


def build_sub_model_from(network: tf.keras.Model, from_layer_name: str):
    """EXPERIMENTAL: NOT GUARANTEED TO WORK WITH ALL ARCHITECTURES.

    Build a sub model of a given network that has inputs at layers defined by a given name.
    Outputs will remain the network outputs."""
    all_layer_names = [l.name for l in network.layers]

    first_layer_index = all_layer_names.index(from_layer_name)
    first_layer = network.layers[first_layer_index]

    new_input = tf.keras.layers.Input(first_layer.input_shape[1:])
    x = first_layer(new_input)
    for layer in network.layers[first_layer_index + 1:]:
        x = layer(x)

    return tf.keras.Model(inputs=new_input, outputs=x)


def get_component(model: tf.keras.Model, name: str):
    """Get outer layer/component by name."""
    for layer in model.layers:
        layer_name = layer.name
        if isinstance(layer, TimeDistributed):
            layer_name = layer.layer.name

        if layer_name == name:
            return layer


def reset_states_masked(model: tf.keras.Model, mask: List):
    """Reset a stateful model's states only at the samples in the batch that are specified by the mask.

    The mask should be a list of length 'batch size' and contain one at every position where the state should be reset,
    and zeros otherwise (booleans possible too)."""

    # extract recurrent layers by their superclass RNN
    recurrent_layers = [layer for layer in model.submodules if isinstance(layer, tf.keras.layers.RNN)]

    for layer in recurrent_layers:
        current_states = [state.numpy() for state in layer.states]
        initial_states = 0
        new_states = []
        for current_state in current_states:
            expanded_mask = numpy.tile(numpy.rot90(numpy.expand_dims(mask, axis=0)), (1, current_state.shape[-1]))
            masked_reset_state = np.where(expanded_mask, initial_states, current_state)
            new_states.append(masked_reset_state)

        layer.reset_states(new_states)


def calc_max_memory_usage(model: tf.keras.Model):
    """Calculate memory requirement of a model per sample in bits."""
    layers = model.submodules
    n_shapes = int(numpy.sum(
        [numpy.prod(numpy.array([s if isinstance(s, int) else 1 for s in l.output_shape])) for l in layers]))
    n_parameters = model.count_params()

    # memory needed for saving activations during gradient calculation
    n_activations = 0
    for l in layers:
        if len(l.trainable_variables) == 0 or l.output_shape is None:
            continue

        activation_shapes = l.output_shape
        if not isinstance(activation_shapes[0], tuple):
            activation_shapes = [tuple(activation_shapes)]

    total_memory = (n_shapes + n_parameters + n_activations) * 32

    return total_memory * 1.1641532182693481 * 10 ** -10





CONVOLUTION_BASE_CLASS = tf.keras.layers.Conv2D.__bases__[0]


def is_conv(layer):
    """Check if layer is convolutional."""
    return isinstance(layer, CONVOLUTION_BASE_CLASS)


# Validators

def validate_model_builder(model_function: Callable, env: BaseWrapper, distribution: BasePolicyDistribution) -> None:
    models = model_function(env, distribution)

    # validate model order
    if not models[1].output_shape[-1] == 1:
        raise ValueError("The model function returns a model tuple whose value network's output is not one-dimensional."
                         "Make sure your value function returns a valid one-dimensional scalar output and that the "
                         "model builder returns this network as the second element of the model-triple.")
    if not models[2].output_shape:
        pass


def validate_env_model_compatibility(env: BaseWrapper, model: tf.keras.Model) -> bool:
    """Validate whether the envs states can be processed by the model."""
    model_inputs = [ip.name for ip in model.inputs]
    env_outputs = list(env.reset()[0].dict().keys())

    if not sorted(model_inputs) == sorted(env_outputs):
        raise IncompatibleModelException(
            f"The model with inputs {model_inputs} cannot handle this environment's states with senses {env_outputs}."
        )

    return True
