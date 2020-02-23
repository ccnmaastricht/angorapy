from LSTM.fixedpointfinder import Adamfixedpointfinder
from LSTM.three_bit_flip_flop import Flipflopper
from LSTM.plot_utils import plot_fixed_points, plot_velocities
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

############################################################
# Create and train recurrent model on 3-Bit FlipFop task
############################################################
# specify architecture e.g. 'vanilla' and number of hidden units
from utilities.model_utils import build_sub_model_to

rnn_type = 'lstm'
n_hidden = 24

# initialize Flipflopper class
flopper = Flipflopper(rnn_type=rnn_type, n_hidden=n_hidden)
# generate trials
stim = flopper.generate_flipflop_trials()
# train the model
flopper.train(stim, 2000, save_model=True)
# visualize a single batch after training
# flopper.visualize_flipflop(stim)
# if a trained model has been saved, it may also be loaded
#flopper.load_model()
############################################################
# Initialize fpf and find fixed points
############################################################
# get weights and activations of trained model
#random_seed = 123
#rng = np.random.RandomState(random_seed)
#n = 1024
weights = flopper.model.get_layer(flopper.hps['rnn_type']).get_weights()
activations = flopper.get_activations(stim)
#activations = np.hstack(activations[1:])
#states = activations + 0.1 * rng.randn(*activations.shape)
# lstmcell = build_sub_model_to(flopper.model, [flopper.hps['rnn_type']])
#init_c_h = states
# input = np.vstack(stim['inputs'][:20, :, :])
#input = np.zeros((n, 3))
#c = init_c_h[0:n, n_hidden:]  # [n_batch x n_dims]
#h = init_c_h[0:n, :n_hidden]  # [n_batch x n_dims]

#lstmcell = tf.keras.layers.LSTMCell(n_hidden)

#inputs = tf.constant(input, dtype='float32')

#tuple = tf.Variable([tf.convert_to_tensor(h), tf.convert_to_tensor(c)],
                 #   dtype='float32')

#output, next_state = lstmcell(inputs, tuple)
#lstmcell.set_weights(weights)
#lstmcell.trainable = False
# tuple = tf.reshape(tuple, [2, 2*n_hidden])
# F = tf.concat((next_state[0], next_state[1]), axis=1)

#optimizer = tf.keras.optimizers.Adam(0.001)
#for i in range(5000):
#    with tf.GradientTape() as tape:
 #       q = 0.5 * tf.reduce_sum(tf.square(next_state - tuple), axis=1)
 #       q_scalar = tf.reduce_mean(q)
 #       gradients = tape.gradient(q_scalar, [tuple])
 #   optimizer.apply_gradients(zip(gradients, [tuple]))

  #  print(q_scalar.numpy())
  #  print(tf.reduce_mean(tuple))
#jac = tape.jacobian(q, [tuple])
# initialize adam fpf
fpf = Adamfixedpointfinder(weights, rnn_type,
                           q_threshold=1e-05,
                           epsilon=0.01,
                           alr_decayr=0.001,
                           max_iters=5000)
# sample states, i.e. a number of ICs
states = fpf.sample_states(activations, 500, 2)
# vel = fpf.compute_velocities(np.hstack(activations[1:]), np.zeros((32768, 3)))
# generate corresponding input as zeros for flip flop task
# please keep in mind that the input does not need to be zero for all tasks
inputs = np.zeros((states.shape[0], 3))
# find fixed points
fps = fpf.find_fixed_points(states, inputs)
# plot fixed points and state trajectories in 3D
if rnn_type == 'lstm':
    activation = np.hstack(activations[1:])
plot_fixed_points(activation, fps, 3000, 4)
# plot_velocities(activation, vel, 3000)



import sklearn.decomposition as skld
#import matplotlib.pyplot as plt
# q_true = q.numpy() < 1e-12

#n_points = 10000
#pca = skld.PCA(3)
#pca.fit(activations)
#X_pca = pca.transform(activations)
#new_pca = pca.transform(np.concatenate(tuple.numpy(), axis=1))

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.plot(X_pca[:n_points, 0], X_pca[:n_points, 1], X_pca[:n_points, 2],
  #      linewidth=0.2)

#ax.scatter(new_pca[:, 0], new_pca[:, 1], new_pca[:, 2],
 #          marker='.', s=30, c='k')

#ax.set_xlabel('PC1')
#ax.set_ylabel('PC2')
#ax.set_zlabel('PC3')
#plt.show()