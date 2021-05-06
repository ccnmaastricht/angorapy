import tensorflow as tf

sequence_length = 10
bs = 1
n_features = 64
layer_sizes = (32, 32)

inputs = tf.keras.Input(batch_shape=(bs, sequence_length, n_features), name="input_layer")
masked = tf.keras.layers.Masking(batch_input_shape=(bs, sequence_length, n_features))(inputs)

# build encoder model
encoder_inputs = tf.keras.Input(shape=(n_features, ), batch_size=bs * sequence_length, name=f"encoder_input")
encoded_timestep = encoder_inputs
for i in range(len(layer_sizes)):
    encoded_timestep = tf.keras.layers.Dense(layer_sizes[i],
                                             kernel_initializer=tf.keras.initializers.Orthogonal(gain=tf.sqrt(2.0)),
                                             bias_initializer=tf.constant_initializer(0.0))(encoded_timestep)
    encoded_timestep = tf.keras.layers.Activation("relu")(encoded_timestep)

encoder_sub_model = tf.keras.Model(inputs=encoder_inputs, outputs=encoded_timestep, name="encoder")

# encoder_sub_model = tf.keras.Sequential([
#     tf.keras.Input(shape=(n_features, ), batch_size=bs * sequence_length, name=f"encoder_input"),
#     tf.keras.layers.Dense(100),
#     tf.keras.layers.Dense(100),
#     tf.keras.layers.Dense(100),
# ])

# encoder_sub_model = tf.keras.layers.Dense(100)

# build recurrent part
x = tf.keras.layers.TimeDistributed(encoder_sub_model, name="td_encoder")(masked)
tded = x
x.set_shape([bs] + x.shape[1:])

x, *_ = tf.keras.layers.GRU(layer_sizes[-1],
                            stateful=True,
                            return_sequences=True,
                            return_state=True,
                            batch_size=bs,
                            name="recurrent_layer")(x)

out = tf.keras.layers.Dense(3)(x)

model = tf.keras.Model(inputs=inputs, outputs=out, name="simple_rnn")
print("AH")