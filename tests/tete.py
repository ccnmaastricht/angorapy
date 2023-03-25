import functools
import os
from typing import Callable, List, Optional

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# NESTED
inp_inner_inner = tf.keras.Input(shape=(2,))
x = tf.keras.layers.Dense(4, name="iid_1")(inp_inner_inner)
inner_inner_model = tf.keras.Model(inputs=[inp_inner_inner], outputs=[x], name="inner_model")

inp_1 = tf.keras.Input(shape=(2,))
x = tf.keras.layers.Dense(2, name="id_1")(inp_1)
x = inner_inner_model(x)
inner_model = tf.keras.Model(inputs=[inp_1], outputs=[x], name="inner_model")

inp_outer = tf.keras.Input((4,))
y = tf.keras.layers.Dense(4, name="od_1")(inp_outer)
y = tf.keras.layers.Dense(2, name="od_2")(y)
y = inner_model(y)
y = tf.keras.layers.Dense(10, name="od_3")(y)
y = tf.keras.layers.Dense(10, name="od_4")(y)
final_model = tf.keras.Model(inputs=[inp_outer], outputs=[y])


def proxy_call(input: tf.Tensor, obj: tf.keras.layers.Layer, **kwargs) -> tf.Tensor:
    if obj._before_call is not None:
        obj._before_call(obj, input)
    output = obj._old_call(input)
    if obj._after_call is not None:
        hook_result = obj._after_call(obj, input, output)
        if hook_result is not None:
            output = hook_result
    return output


def hook_layer_call(layers: List[tf.keras.layers.Layer],
                    before_call: Callable[[tf.keras.layers.Layer, tf.Tensor], None] = None,
                    after_call: Callable[[tf.keras.layers.Layer, tf.Tensor, tf.Tensor], Optional[tf.Tensor]] = None):
    for layer in layers:
        layer._before_call = before_call
        layer._after_call = after_call
        layer._old_call = layer.call
        layer.call = functools.partial(proxy_call, obj=layer)


def print_input_output(layer: tf.keras.layers.Layer, input: tf.Tensor, output: tf.Tensor):
    print(input, output)


hook_layer_call(final_model.layers, after_call=print_input_output)

final_model(tf.random.normal((1, 4)))
