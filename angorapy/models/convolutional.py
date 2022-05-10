#!/usr/bin/env python
"""Convolutional components/networks."""

import tensorflow as tf


def _residual_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    """A residual block.

    Arguments:
      x: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.
      name: string, block label.

    Returns:
      Output tensor for the residual block.
    """
    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut:
        # linear convolutional shortcut that matches the sizes
        shortcut = tf.keras.layers.Conv2D(4 * filters, 1, strides=stride, name=name + '_0_conv')(x)
        shortcut = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = tf.keras.layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = tf.keras.layers.Activation('relu', name=name + '_1_relu')(x)

    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = tf.keras.layers.Activation('relu', name=name + '_2_relu')(x)

    x = tf.keras.layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = tf.keras.layers.Add(name=name + '_add')([shortcut, x])
    x = tf.keras.layers.Activation('relu', name=name + '_out')(x)

    return x


# VISUAL ENCODING

def _build_openai_encoder(shape, out_shape, name="visual_component", batch_size=None):
    """Shallow AlexNet Version. Original number of channels are too large for normal GPU memory.

    Args:
        out_shape:
    """
    inputs = tf.keras.Input(batch_shape=(batch_size,) + tuple(shape))

    # first layer
    x = tf.keras.layers.Conv2D(32, 5, 1, padding="valid")(inputs)  # 196 x 196
    x = tf.keras.layers.Activation("relu")(x)

    # second layer
    x = tf.keras.layers.Conv2D(32, 3, 1, padding="valid")(x)  # 195 x 195
    x = tf.keras.layers.Activation("relu")(x)

    # pooling
    x = tf.keras.layers.MaxPool2D(3, 3)(x)  # 64 x 64

    # resnet
    x = _residual_block(x, 16, 3, 3, name="res1")
    x = _residual_block(x, 32, 3, 3, name="res2")
    x = _residual_block(x, 64, 3, 3, name="res3")
    x = _residual_block(x, 64, 3, 3, name="res4")

    # fully connected
    x = tf.keras.layers.Activation("softmax")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.Activation("relu")(x)

    # output
    x = tf.keras.layers.Dense(out_shape)(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name=name)


def _build_openai_small_encoder(shape, out_shape, name="visual_component", batch_size=None):
    """Shallow AlexNet Version. Original number of channels are too large for normal GPU memory.

    Args:
        out_shape:
    """
    inputs = tf.keras.Input(shape=shape, batch_size=batch_size, name=name + "_input")

    # first layer
    x = tf.keras.layers.Conv2D(32, 5, 1, padding="valid")(inputs)  # 96 x 96
    x = tf.keras.layers.Activation("relu")(x)

    # second layer
    x = tf.keras.layers.Conv2D(32, 3, 1, padding="valid")(x)  # 94 x 94
    x = tf.keras.layers.Activation("relu")(x)

    # pooling
    x = tf.keras.layers.MaxPool2D(3, 3)(x)  # 31 x 31

    # resnet
    x = _residual_block(x, 16, 3, 3, name="res1")
    x = _residual_block(x, 32, 3, 3, name="res2")
    x = _residual_block(x, 64, 3, 3, name="res3")

    # fully connected
    x = tf.keras.layers.Activation("softmax")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.L2(l2=0.001))(x)
    x = tf.keras.layers.Activation("relu")(x)

    # output
    x = tf.keras.layers.Dense(out_shape)(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name=name)


def _build_alexnet_encoder(shape, batch_size=None, name="visual_component"):
    """Shallow AlexNet Version. Original number of channels are too large for normal GPU memory."""
    inputs = tf.keras.Input(batch_shape=(batch_size,) + tuple(shape))

    # first layer
    x = tf.keras.layers.Conv2D(32, 11, 4)(inputs)
    x = tf.keras.layers.Activation("tanh")(x)
    x = tf.keras.layers.MaxPool2D(3, 2)(x)

    # second layer
    x = tf.keras.layers.Conv2D(32, 5, 1, padding="same")(x)
    x = tf.keras.layers.Activation("tanh")(x)
    x = tf.keras.layers.MaxPool2D(3, 2)(x)

    # third layer
    x = tf.keras.layers.Conv2D(64, 3, 1, padding="same")(x)
    x = tf.keras.layers.Activation("tanh")(x)

    # fourth layer
    x = tf.keras.layers.Conv2D(64, 3, 1, padding="same")(x)
    x = tf.keras.layers.Activation("tanh")(x)

    # fifth layer
    x = tf.keras.layers.Conv2D(32, 3, 1, padding="same")(x)
    x = tf.keras.layers.Activation("tanh")(x)

    x = tf.keras.layers.MaxPool2D(3, 2)(x)

    # fully connected layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.Activation("tanh")(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.Activation("tanh")(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name=name)


# VISUAL DECODING

def _build_visual_decoder(input_dim):
    model = tf.keras.Sequential((
        tf.keras.layers.Dense(512, input_dim=input_dim, activation="relu"),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(1152, activation="relu"),

        tf.keras.layers.Reshape((6, 6, 32)),
        tf.keras.layers.Conv2DTranspose(32, 3, 2),
        tf.keras.layers.Conv2DTranspose(64, 3, 1, activation="relu", padding="same"),
        tf.keras.layers.Conv2DTranspose(64, 3, 1, activation="relu", padding="same"),
        tf.keras.layers.Conv2DTranspose(32, 3, 2),
        tf.keras.layers.Conv2DTranspose(32, 5, 1, activation="relu", padding="same"),
        tf.keras.layers.Conv2DTranspose(32, 3, 2),
        tf.keras.layers.Conv2DTranspose(3, 11, 4, activation="sigmoid"),
    ))

    model.build()
    return model


if __name__ == "__main__":
    conv_comp = _build_openai_small_encoder((100, 100, 3), 7)
    print(conv_comp(tf.random.normal((1, 100, 100, 3))).shape)
    conv_comp.summary()
    #
    # print("\n\n\n")
    # conv_comp = _build_openai_encoder((200, 200, 3), 7)
    # conv_comp.summary()
