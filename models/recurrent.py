#!/usr/bin/env python
"""TODO Module Docstring."""
import os

import gym
import tensorflow as tf
from gym.spaces import Box
from tensorflow.keras.layers import TimeDistributed as TD
<<<<<<< HEAD
from tensorflow.keras import utils
# import keras
import numpy as np
import random
from collections import deque
from agent.policy import act_continuous

from tqdm import tqdm


from models.components import _build_encoding_sub_model, _build_continuous_head, _build_discrete_head
from utilities.util import env_extract_dims


def build_rnn_distinct_models(env: gym.Env, bs: int = 1):
    """Build simple policy and value models having an LSTM before their heads."""
    continuous_control = isinstance(env.action_space, Box)
    state_dimensionality, n_actions = env_extract_dims(env)

    inputs = tf.keras.Input(batch_shape=(bs, None, state_dimensionality,))
    masked = tf.keras.layers.Masking()(inputs)

    # policy network
    x = TD(_build_encoding_sub_model((state_dimensionality,), bs, layer_sizes=(64,), name="policy_encoder"),
           name="TD_policy")(masked)
    x.set_shape([bs] + x.shape[1:])
    x = tf.keras.layers.SimpleRNN(64, stateful=True, return_sequences=True, batch_size=bs,
                                  name="policy_recurrent_layer")(x)

    if continuous_control:
        out_policy = _build_continuous_head(n_actions, x.shape[1:], bs)(x)
    else:
        out_policy = _build_discrete_head(n_actions, x.shape[1:], bs)(x)

    # value network
    x = TD(_build_encoding_sub_model((state_dimensionality,), bs, layer_sizes=(64,), name="value_encoder"),
           name="TD_value")(masked)
    x.set_shape([bs] + x.shape[1:])
    x = tf.keras.layers.SimpleRNN(64, stateful=True, return_sequences=True, batch_size=bs,
                                  name="value_recurrent_layer")(x)

    out_value = tf.keras.layers.Dense(1)(x)

    policy = tf.keras.Model(inputs=inputs, outputs=out_policy, name="simple_rnn_policy")
    value = tf.keras.Model(inputs=inputs, outputs=out_value, name="simple_rnn_value")

<<<<<<< HEAD
    return policy, value, tf.keras.Model(inputs=inputs, outputs=[out_policy, out_value], name="policy_value")
# hyperparameters for lunarlander


GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 10000
BATCH_SIZE = 3

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995


class RecurrentLander:

    def __init__(self):
        # self.exploration_rate = EXPLORATION_MAX
        # self.observation_space = observation_space
        # self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.batch_size = BATCH_SIZE

    # def act(self, pv, state):
    # if np.random.rand() < self.exploration_rate:

        # else:
            # out_pi, out_v = pv.predict(state)
            # return out_pi, out_v

    def gather(self, pi, environment):
        state = environment.reset()
        environment.render()
        terminal = False
        for l in range(120):
            while not terminal:
                policy_out = pi.predict(tf.random.normal((1, 1, 8)))
                action = act_continuous(policy_out)
                state_next, reward, terminal, info = environment.step(action)
                self.memory.append((state, action, state_next, reward, terminal))
                state = state_next

    def learn(self, pi):
        batch = np.array(random.sample(self.memory, self.batch_size))
        batch = np.array(batch)
        for state, action, reward, state_next, terminal, value_out in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(pi.predict(state_next)[0]))
            q_values = pi.predict(state)
            q_values[0][action] = q_update
            pi.model.fit(state, q_values, verbose=0)

        return pv
=======
    return policy, value, tf.keras.Model(inputs=inputs, outputs=[out_policy, out_value], name="simple_rnn")
>>>>>>> master


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
<<<<<<< HEAD
    lander = RecurrentLander()
    environment = gym.make("LunarLanderContinuous-v2")
    pi, v, pv = build_rnn_distinct_models(environment, bs=3)

    utils.plot_model(pi)

    lander.gather(pi, environment)
    pi = lander.learn(pi)

    # out_pi, out_v = pv.predict(tf.random.normal((3, 16, 8)))
    # print(out_pi.shape)
    # print(out_v.shape)
=======

    environment = gym.make("CartPole-v1")
    pi, v, pv = build_rnn_distinct_models(environment, bs=3)

    tf.keras.utils.plot_model(pv, expand_nested=True)

    for i in tqdm(range(10)):
        out_pi, out_v = pv.predict(tf.random.normal((3, 16, 4)))
>>>>>>> master
