#!/usr/bin/env python
"""Hybrid policy networks that utilize both visual and unstructured input data."""
import os
import time
from typing import Iterable

import tensorflow as tf
from gym.spaces import Box
from tensorflow.keras.layers import TimeDistributed as TD
from tensorflow_core.python.keras.utils import plot_model
from tqdm import tqdm

from agent.policies import BasePolicyDistribution, BetaPolicyDistribution
from environments import *
from models.components import _build_fcn_component
from models.convolutional import _build_visual_encoder
from utilities.const import VISION_WH
from utilities.util import env_extract_dims
from utilities.model_management import calc_max_memory_usage


def build_shadow_brain_v1(env: gym.Env, distribution: BasePolicyDistribution, bs: int):
    """Build network for the shadow hand task."""
    state_dimensionality, n_actions = env_extract_dims(env)
    hidden_dimensions = 32

    # inputs
    visual_in = tf.keras.Input(batch_shape=(bs, None, VISION_WH, VISION_WH, 3), name="visual_input")
    proprio_in = tf.keras.Input(batch_shape=(bs, None, 48,), name="proprioceptive_input")
    touch_in = tf.keras.Input(batch_shape=(bs, None, 92,), name="somatosensory_input")
    goal_in = tf.keras.Input(batch_shape=(bs, None, 7,), name="goal_input")

    # abstractions of perceptive inputs
    visual_latent = TD(_build_visual_encoder(shape=(VISION_WH, VISION_WH, 3), batch_size=bs))(visual_in)
    proprio_latent = TD(_build_fcn_component(48, 12, 8, batch_size=bs, name="latent_proprio"))(proprio_in)
    touch_latent = TD(_build_fcn_component(92, 24, 8, batch_size=bs, name="latent_touch"))(touch_in)

    # concatenation of perceptive abstractions
    concatenation = tf.keras.layers.Concatenate()([visual_latent, proprio_latent, touch_latent])

    # fully connected layer integrating perceptive representations
    x = TD(tf.keras.layers.Dense(48))(concatenation)
    x = TD(tf.keras.layers.ReLU())(x)

    # concatenation of goal and perception
    x.set_shape([bs] + x.shape[1:])
    x = tf.keras.layers.Concatenate()([x, goal_in])

    # recurrent layer
    o = tf.keras.layers.GRU(hidden_dimensions, stateful=True, return_sequences=True, batch_size=bs)(x)

    # output heads
    policy_out = distribution.build_action_head(n_actions, o.shape[1:], bs)(o)
    value_out = tf.keras.layers.Dense(1, name="value")(o)

    # define models
    policy = tf.keras.Model(inputs=[visual_in, proprio_in, touch_in, goal_in],
                            outputs=[policy_out], name="shadow_brain_v1_policy")
    value = tf.keras.Model(inputs=[visual_in, proprio_in, touch_in, goal_in],
                           outputs=[value_out], name="shadow_brain_v1_value")
    joint = tf.keras.Model(inputs=[visual_in, proprio_in, touch_in, goal_in],
                           outputs=[policy_out, value_out], name="shadow_brain_v1")

    return policy, value, joint


def build_blind_shadow_brain_v1(env: gym.Env, distribution: BasePolicyDistribution, bs: int):
    """Build network for the shadow hand task but without visual inputs."""
    state_dimensionality, n_actions = env_extract_dims(env)
    hidden_dimensions = 32

    # inputs
    object_in = tf.keras.Input(batch_shape=(bs, None, 13), name="object_pos_input")
    proprio_in = tf.keras.Input(batch_shape=(bs, None, 48,), name="proprioceptive_input")
    touch_in = tf.keras.Input(batch_shape=(bs, None, 92,), name="somatosensory_input")
    goal_in = tf.keras.Input(batch_shape=(bs, None, 7,), name="goal_input")

    # abstractions of perceptive inputs
    object_latent = TD(_build_fcn_component(13, 12, 8, batch_size=bs, name="latent_object_pos"))(object_in)
    proprio_latent = TD(_build_fcn_component(48, 12, 8, batch_size=bs, name="latent_proprio"))(proprio_in)
    touch_latent = TD(_build_fcn_component(92, 24, 8, batch_size=bs, name="latent_touch"))(touch_in)

    # concatenation of perceptive abstractions
    concatenation = tf.keras.layers.Concatenate()([object_latent, proprio_latent, touch_latent])

    # fully connected layer integrating perceptive representations
    x = TD(tf.keras.layers.Dense(48))(concatenation)
    x = TD(tf.keras.layers.ReLU())(x)

    # concatenation of goal and perception
    x.set_shape([bs] + x.shape[1:])
    x = tf.keras.layers.Concatenate()([x, goal_in])

    # recurrent layer
    o = tf.keras.layers.SimpleRNN(hidden_dimensions, stateful=True, return_sequences=True, batch_size=bs)(x)

    # output heads
    policy_out = distribution.build_action_head(n_actions, o.shape[1:], bs)(o)
    value_out = tf.keras.layers.Dense(1, name="value")(o)

    # define models
    policy = tf.keras.Model(inputs=[object_in, proprio_in, touch_in, goal_in],
                            outputs=[policy_out], name="blind_shadow_brain_v1_policy")
    value = tf.keras.Model(inputs=[object_in, proprio_in, touch_in, goal_in],
                           outputs=[value_out], name="blind_shadow_brain_v1_value")
    joint = tf.keras.Model(inputs=[object_in, proprio_in, touch_in, goal_in],
                           outputs=[policy_out, value_out], name="blind_shadow_brain_v1")

    return policy, value, joint


def build_shadow_brain_v2(env: gym.Env, distribution: BasePolicyDistribution, bs: int):
    """Build network for the shadow hand task, version 2."""
    state_dimensionality, n_actions = env_extract_dims(env)
    hidden_dimensions = 32

    # inputs
    visual_in = tf.keras.Input(batch_shape=(bs, None, VISION_WH, VISION_WH, 3), name="visual_input")
    proprio_in = tf.keras.Input(batch_shape=(bs, None, 48,), name="proprioceptive_input")
    touch_in = tf.keras.Input(batch_shape=(bs, None, 92,), name="somatosensory_input")
    goal_in = tf.keras.Input(batch_shape=(bs, None, 7,), name="goal_input")

    # abstractions of perceptive inputs
    visual_latent = TD(_build_visual_encoder(shape=(VISION_WH, VISION_WH, 3), batch_size=bs))(visual_in)
    visual_latent = TD(tf.keras.layers.Dense(128))(visual_latent)
    visual_latent = TD(tf.keras.layers.ReLU())(visual_latent)
    visual_latent.set_shape([bs] + visual_latent.shape[1:])
    visual_plus_goal = tf.keras.layers.Concatenate()([visual_latent, goal_in])
    eigengrasps = TD(tf.keras.layers.Dense(20))(visual_plus_goal)
    eigengrasps = TD(tf.keras.layers.ReLU())(eigengrasps)

    # concatenation of touch and proprioception
    proprio_touch = tf.keras.layers.Concatenate()([proprio_in, touch_in])
    proprio_touch_latent = TD(tf.keras.layers.Dense(20))(proprio_touch)
    proprio_touch_latent = TD(tf.keras.layers.ReLU())(proprio_touch_latent)

    # concatenation of goal and perception
    proprio_touch_latent.set_shape([bs] + proprio_touch_latent.shape[1:])
    eigengrasps.set_shape([bs] + eigengrasps.shape[1:])
    x = tf.keras.layers.Concatenate()([goal_in, eigengrasps, proprio_touch_latent])

    # recurrent layer
    rnn_out = tf.keras.layers.GRU(hidden_dimensions, stateful=True, return_sequences=True, batch_size=bs)(x)

    # output heads
    policy_out = distribution.build_action_head(n_actions, rnn_out.shape[1:], bs)(rnn_out)
    value_out = tf.keras.layers.Dense(1, name="value")(rnn_out)

    # define models
    policy = tf.keras.Model(inputs=[visual_in, proprio_in, touch_in, goal_in],
                            outputs=[policy_out], name="shadow_brain_v2_policy")
    value = tf.keras.Model(inputs=[visual_in, proprio_in, touch_in, goal_in],
                           outputs=[value_out], name="shadow_brain_v2_value")
    joint = tf.keras.Model(inputs=[visual_in, proprio_in, touch_in, goal_in],
                           outputs=[policy_out, value_out], name="shadow_brain_v2")

    return policy, value, joint


if __name__ == "__main__":

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    sequence_length = 100
    batch_size = 256

    env = gym.make("ShadowHand-v0")
    _, _, joint = build_shadow_brain_v2(env, BetaPolicyDistribution(), bs=batch_size)
    plot_model(joint, to_file=f"{joint.name}.png", expand_nested=True, show_shapes=True)
    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.SGD()


    @tf.function
    def _train():
        start_time = time.time()

        for _ in tqdm(range(sequence_length), disable=False):
            sample_batch = (tf.convert_to_tensor(tf.random.normal([batch_size, 4, VISION_WH, VISION_WH, 3])),
                            tf.convert_to_tensor(tf.random.normal([batch_size, 4, 48])),
                            tf.convert_to_tensor(tf.random.normal([batch_size, 4, 92])),
                            tf.convert_to_tensor(tf.random.normal([batch_size, 4, 7])))

            with tf.GradientTape() as tape:
                out, v = joint(sample_batch, training=True)
                loss = tf.reduce_mean(out - v)

            grads = tape.gradient(loss, joint.trainable_variables)
            optimizer.apply_gradients(zip(grads, joint.trainable_variables))

        print(f"Execution Time: {time.time() - start_time}")


    _train()
