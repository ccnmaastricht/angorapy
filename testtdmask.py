import gym
import tensorflow as tf
from tensorflow.python.keras.layers import TimeDistributed

shared = False
bs = 1
model_type = "rnn"
layer_sizes = (64, 64)

state_dimensionality, n_actions = 8, 4
rnn_choice = tf.keras.layers.SimpleRNN

sequence_length = None
inputs = tf.keras.Input(shape=(sequence_length, state_dimensionality,), batch_size=bs)
masked = tf.keras.layers.Masking()(inputs)

# build encoder
encoder_inputs = tf.keras.Input(shape=state_dimensionality, batch_size=bs)
encoder_x = encoder_inputs
for i in range(len(layer_sizes)):
    encoder_x = tf.keras.layers.Dense(layer_sizes[i],
                                      kernel_initializer=tf.keras.initializers.Orthogonal(gain=tf.sqrt(2.0)),
                                      bias_initializer=tf.constant_initializer(0.0))(encoder_x)
    encoder_x = tf.keras.layers.Activation("tanh")(encoder_x)

encoder = tf.keras.Model(inputs=encoder_inputs, outputs=encoder_x, name="policy_encoder")

# policy network; stateful, so batch size needs to be known
x = TimeDistributed(encoder, name="TD_policy")(masked)
x.set_shape([bs] + x.shape[1:])

x, *_ = rnn_choice(layer_sizes[-1],
                   stateful=True,
                   return_sequences=True,
                   return_state=True,
                   batch_size=bs,
                   name="policy_recurrent_layer")(x)

means = tf.keras.layers.Dense(n_actions, name="means",
                              kernel_initializer=tf.keras.initializers.Orthogonal(0.01),
                              bias_initializer=tf.keras.initializers.Constant(0.0))(x)

policy = tf.keras.Model(inputs=inputs, outputs=means, name="simple_rnn_policy")
