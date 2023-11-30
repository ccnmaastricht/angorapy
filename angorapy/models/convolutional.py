#!/usr/bin/env python
"""Convolutional components/networks."""
import kortex
import tensorflow as tf
import tensorflow_models as tfm


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
    x = tfm.vision.layers.ResidualBlock(
        filters,
        strides=stride1,
        name=name + "_block1",
        use_projection=True,
        kernel_regularizer=tf.keras.regularizers.L2(0.001),
        bias_regularizer=tf.keras.regularizers.L2(0.001),
    )(x)

    for i in range(2, blocks + 1):
        x = tfm.vision.layers.ResidualBlock(
            filters,
            strides=1,
            use_projection=False,
            name=name + "_block" + str(i),
            kernel_regularizer=tf.keras.regularizers.L2(0.001),
            bias_regularizer=tf.keras.regularizers.L2(0.001),
        )(x)
    return x


def _build_openai_resnets(shape, batch_size=None):
    """Shallow AlexNet Version. Original number of channels are too large for normal GPU memory.

    """
    inputs = tf.keras.Input(batch_shape=(batch_size,) + tuple(shape))

    # first layer
    x = tf.keras.layers.Conv2D(32, 5, 1, padding="valid",
                               kernel_regularizer=tf.keras.regularizers.L2(0.001),
                               bias_regularizer=tf.keras.regularizers.L2(0.001),
                               )(inputs)  # 196 x 196
    x = tf.keras.layers.Activation("relu")(x)

    # second layer
    x = tf.keras.layers.Conv2D(32, 3, 1, padding="valid",
                               kernel_regularizer=tf.keras.regularizers.L2(0.001),
                               bias_regularizer=tf.keras.regularizers.L2(0.001),
                               )(x)  # 195 x 195
    x = tf.keras.layers.Activation("relu")(x)

    # pooling
    x = tf.keras.layers.MaxPool2D(3, 3)(x)  # 64 x 64

    # resnet
    x = _residual_stack(x, filters=16, blocks=1, stride1=1, name="res1")
    x = _residual_stack(x, filters=32, blocks=2, stride1=3, name="res2")
    x = _residual_stack(x, filters=64, blocks=2, stride1=3, name="res3")
    x = _residual_stack(x, filters=64, blocks=2, stride1=3, name="res4")

    return tf.keras.Model(inputs=inputs, outputs=x, name="resnetblocks")


# VISUAL ENCODING


class OpenAIEncoder(tf.keras.Model):

    def __init__(self, shape, n_cameras=1, **kwargs):
        super().__init__(**kwargs)

        self.n_cameras = n_cameras

        self.rescale = tf.keras.layers.Rescaling(1.0 / 255)
        self.resnet_encoder = _build_openai_resnets(shape=shape,
                                                    batch_size=kwargs.get("batch_size"))

        self.pose_network = tf.keras.Sequential([
            tf.keras.layers.Dense(
                256,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(0.001),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(
                256,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(0.001),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(
                128,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.L2(0.001),
            ),
        ])

        self.relu = tf.keras.layers.Activation("relu")
        self.softmax = kortex.layers.SpatialSoftargmax()

        self.pos_dense = tf.keras.layers.Dense(
            3,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=lambda shape, dtype: tf.convert_to_tensor([0.32801157, 0.00065984, 0.02937366]),
            kernel_regularizer=tf.keras.regularizers.L2(0.001),
            bias_regularizer=tf.keras.regularizers.L2(0.001)
        )
        self.rot_dense = tf.keras.layers.Dense(
            4,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=lambda shape, dtype: tf.convert_to_tensor([0.6211242, -0.02646778, 0.02921998, -0.00490309]),
            kernel_regularizer=tf.keras.regularizers.L2(0.001),
            bias_regularizer=tf.keras.regularizers.L2(0.001)
        )

    def call(self, inputs, training=None, mask=None):
        per_camera_output = []

        inputs = self.rescale(inputs)
        inputs = tf.split(inputs, num_or_size_splits=self.n_cameras, axis=-1)

        for camera in inputs:
            x = self.resnet_encoder(camera)
            x = self.softmax(x)
            per_camera_output.append(x)

        # fully connected
        if len(per_camera_output) > 1:
            x = tf.keras.layers.concatenate(per_camera_output, axis=-1)
        else:
            x = per_camera_output[0]

        x = self.pose_network(x)

        # output
        pos = self.pos_dense(x)
        rot = self.rot_dense(x)
        rot = tf.math.l2_normalize(rot, axis=-1)  # ensure quaternions are quaternions

        outputs = tf.keras.layers.concatenate([pos, rot], axis=-1)

        return outputs


if __name__ == "__main__":
    conv_comp = OpenAIEncoder((64, 64, 3), 1, mode="classification")
    print(conv_comp(tf.random.normal((1, 64, 64, 3))).shape)
    conv_comp.summary(expand_nested=True, )
