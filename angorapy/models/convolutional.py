#!/usr/bin/env python
"""Convolutional components/networks."""
import keras_cortex.layers
import tensorflow as tf


def _residual_stack(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.
    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the first layer in the first block.
      name: string, stack label.
    Returns:
      Output tensor for the stacked blocks.
    """
    x = tfm.vision.layers.ResidualBlock(filters, strides=stride1, name=name + "_block1", use_projection=True)(x)
    for i in range(2, blocks + 1):
        x = tfm.vision.layers.ResidualBlock(
            filters, strides=1, use_projection=False, name=name + "_block" + str(i)
        )(x)
    return x


# VISUAL ENCODING


class OpenAIEncoder(tf.keras.Model):

    def __init__(self, shape, n_cameras=1, mode="pose", **kwargs):
        super().__init__(**kwargs)

        assert mode in ["pose", "classification"]
        self.mode = mode

        self.n_cameras = n_cameras

        self.rescale = tf.keras.layers.Rescaling(1.0 / 255)
        self.resnet_encoder = _build_openai_resnets(shape=shape,
                                                    batch_size=kwargs.get("batch_size"))

        self.dense = tf.keras.layers.Dense(128)
        self.relu = tf.keras.layers.Activation("relu")
        self.flatten = tf.keras.layers.Flatten()
        self.softmax = keras_cortex.layers.SpatialSoftmax()

        if mode == "pose":
            self.pos_dense = tf.keras.layers.Dense(3, kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                                   bias_initializer=tf.keras.initializers.Constant(0),
                                                   kernel_regularizer=tf.keras.regularizers.L2(0.001),
                                                   bias_regularizer=tf.keras.regularizers.L2(0.001))
            self.rot_dense = tf.keras.layers.Dense(4, kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                                   bias_initializer=tf.keras.initializers.Constant(0),
                                                   kernel_regularizer=tf.keras.regularizers.L2(0.001),
                                                   bias_regularizer=tf.keras.regularizers.L2(0.001))
        elif mode == "classification":
            self.data_augmentation = tf.keras.Sequential(
                [
                    tf.keras.layers.RandomFlip("horizontal"),
                    tf.keras.layers.RandomRotation(0.1),
                ]
            )

            self.classifier = tf.keras.layers.Dense(1000, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        per_camera_output = []

        inputs = self.rescale(inputs)
        inputs = tf.split(inputs, num_or_size_splits=self.n_cameras, axis=-1)

        for camera in inputs:

            if self.mode == "classification":
                camera = self.data_augmentation(camera, training=training)

            x = self.resnet_encoder(camera)
            x = self.softmax(x)
            x = self.flatten(x)
            per_camera_output.append(x)

        # fully connected
        if len(per_camera_output) > 1:
            x = tf.keras.layers.concatenate(per_camera_output, axis=-1)
        else:
            x = per_camera_output[0]
        x = self.dense(x)
        x = self.relu(x)

        # output
        if self.mode == "pose":
            pos = self.pos_dense(x)
            rot = self.rot_dense(x)
            rot = tf.math.l2_normalize(rot, axis=-1)  # ensure quaternions are quaternions

            outputs = tf.keras.layers.concatenate([pos, rot], axis=-1)
        elif self.mode == "classification":
            outputs = self.classifier(x)
        else:
            raise NotImplementedError("Unknown mode: {}".format(self.mode))

        return outputs


def _build_openai_resnets(shape, batch_size=None):
    """Shallow AlexNet Version. Original number of channels are too large for normal GPU memory.

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
    x = _residual_stack(x, filters=16, blocks=1, stride1=1, name="res1")
    x = _residual_stack(x, filters=32, blocks=2, stride1=2, name="res2")
    x = _residual_stack(x, filters=64, blocks=2, stride1=2, name="res3")
    x = _residual_stack(x, filters=64, blocks=2, stride1=2, name="res4")

    return tf.keras.Model(inputs=inputs, outputs=x, name="resnetblocks")


def _build_openai_encoder(shape, name="visual_component", batch_size=None):
    """Shallow AlexNet Version. Original number of channels are too large for normal GPU memory.

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
    x = _residual_stack(x, filters=16, blocks=1, stride1=1, name="res1")
    x = _residual_stack(x, filters=32, blocks=2, stride1=2, name="res2")
    x = _residual_stack(x, filters=64, blocks=2, stride1=2, name="res3")
    x = _residual_stack(x, filters=64, blocks=2, stride1=2, name="res4")

    # fully connected
    # x = keras_cortex.layers.SpatialSoftmax()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.Activation("relu")(x)

    # output
    pos = tf.keras.layers.Dense(3, kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                bias_initializer=tf.keras.initializers.Constant(0),
                                kernel_regularizer=tf.keras.regularizers.L2(0.001),
                                bias_regularizer=tf.keras.regularizers.L2(0.001))(x)
    rot = tf.keras.layers.Dense(4, kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                bias_initializer=tf.keras.initializers.Constant(0),
                                kernel_regularizer=tf.keras.regularizers.L2(0.001),
                                bias_regularizer=tf.keras.regularizers.L2(0.001))(x)
    rot = tf.math.l2_normalize(rot, axis=-1)  # ensure quaternions are quaternions

    outputs = tf.concat([pos, rot], axis=-1)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


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
    conv_comp = OpenAIEncoder((64, 64, 3), 1, mode="classification")
    print(conv_comp(tf.random.normal((1, 64, 64, 3))).shape)
    conv_comp.summary(expand_nested=True, )
    #
    # print("\n\n\n")
    # conv_comp = _build_openai_encoder((200, 200, 3), 7)
    # conv_comp.summary()
