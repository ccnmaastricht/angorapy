import functools
from typing import List, Callable, Optional
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


def register_hook(layers: List[tf.keras.layers.Layer],
                  before_call: Callable[[tf.keras.layers.Layer, tf.Tensor], None] = None,
                  after_call: Callable[[tf.keras.layers.Layer, tf.Tensor, tf.Tensor], Optional[tf.Tensor]] = None):
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