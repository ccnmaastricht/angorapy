import functools
from typing import Callable
from typing import List
from typing import Optional

import tensorflow as tf


def hooked_call(inputs, *args, obj: tf.keras.layers.Layer, **kwargs) -> tf.Tensor:
    if obj._before_call is not None:
        obj._before_call(obj, inputs)
    output = obj._old_call(inputs, *args, **kwargs)
    if obj._after_call is not None:
        hook_result = obj._after_call(obj, inputs, output)
        if hook_result is not None:
            output = hook_result
    return output


def register_hook(
        layers: List[tf.keras.layers.Layer],
        before_call: Callable[[tf.keras.layers.Layer, tf.Tensor], None] = None,
        after_call: Callable[[tf.keras.layers.Layer, tf.Tensor, tf.Tensor], Optional[tf.Tensor]] = None,
):
    """ Register a hook on a list of layers.

    If the hook should receive additional arguments, use 'functools.partial' to create a partial function before passing
    it to this function.

    :param layers:          List of layer objects (Keras) to register the hook on. The function will overwrite the
                            'call' function of these layers
    :param before_call:     Function to call before the original 'call' function of the layer.
    :param after_call:      Function to call after the original 'call' function of the layer.

    :return:                None
    """

    for layer in layers:
        layer._before_call = before_call
        layer._after_call = after_call
        layer._old_call = layer.call
        layer.call = functools.partial(hooked_call, obj=layer)


# todo this could be done through a context
def clear_hooks(module: tf.keras.Model):
    """Remove all hooks from a model."""

    for layer in module.submodules:
        if not hasattr(layer, "_old_call"):
            continue

        del layer._before_call
        del layer._after_call
        layer.call = layer._old_call
        del layer._old_call
