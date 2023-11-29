from typing import Dict

import tensorflow as tf

from angorapy.agent.ppo.optim import learn_on_batch
from angorapy.utilities.model_utils import reset_states_masked_tf
from angorapy.utilities.core import detect_finished_episodes


def ff_train_step(batch,
                  joint,
                  distribution,
                  continuous_control,
                  clip_values,
                  gradient_clipping,
                  clip,
                  c_value,
                  c_entropy,
                  is_recurrent,
                  optimizer):
    grad, ent, pi_loss, v_loss = learn_on_batch(
        batch=batch, joint=joint, distribution=distribution,
        continuous_control=continuous_control, clip_values=clip_values,
        gradient_clipping=gradient_clipping, clipping_bound=clip, c_value=c_value,
        c_entropy=c_entropy, is_recurrent=is_recurrent)
    optimizer.apply_gradients(zip(grad, joint.trainable_variables))

    return ent, pi_loss, v_loss


@tf.function
def _split_batch(batch: Dict[str, tf.Tensor], n_chunks_per_trajectory_per_batch: int):
    _split_part = lambda bv: tf.reshape(bv,
                                       [bv.shape[0]]
                                       + [bv.shape[1] // n_chunks_per_trajectory_per_batch,
                                          n_chunks_per_trajectory_per_batch]
                                       + bv.shape[2:])

    return {k: _split_part(v) for k, v in batch.items()}


@tf.function
def _do_batch_op(joint,
                 recurrent_layers,
                 n_chunks_per_trajectory_per_batch,
                 batch_i,
                 split_batch,
                 distribution,
                 continuous_control,
                 clip_values,
                 gradient_clipping,
                 clip,
                 c_value,
                 c_entropy,
                 is_recurrent):
    batch_grad = [tf.zeros_like(tv) for tv in joint.trainable_variables]
    batch_ent = tf.constant(0.)
    batch_pi_loss = tf.constant(0.)
    batch_v_loss = tf.constant(0.)

    for chunk_i in tf.range(n_chunks_per_trajectory_per_batch):
        partial_batch = {k: v[:, batch_i, chunk_i, ...] for k, v in split_batch.items()}
        grad, ent, pi_loss, v_loss = learn_on_batch(
            batch=partial_batch, joint=joint, distribution=distribution,
            continuous_control=continuous_control, clip_values=clip_values,
            gradient_clipping=gradient_clipping, clipping_bound=clip,
            c_value=c_value, c_entropy=c_entropy, is_recurrent=is_recurrent)

        batch_grad = tf.nest.map_structure(lambda bg, g: tf.add(bg, g), batch_grad, grad)
        reset_mask = detect_finished_episodes(partial_batch["done"])

        batch_ent += ent
        batch_pi_loss += pi_loss
        batch_v_loss += v_loss

        # make partial RNN state resets
        reset_states_masked_tf(recurrent_layers, reset_mask)

    batch_grad = tf.nest.map_structure(lambda b: b / n_chunks_per_trajectory_per_batch, batch_grad)

    return batch_grad, batch_ent, batch_pi_loss, batch_v_loss


def recurrent_train_step(super_batch: dict,
                         batch_size: tf.int32,
                         joint,
                         distribution,
                         continuous_control,
                         clip_values,
                         gradient_clipping,
                         clip,
                         c_value,
                         c_entropy,
                         is_recurrent,
                         optimizer,
                         pbar=None):
    """Recurrent train step, using truncated back propagation through time.

    Incoming batch shape: (BATCH_SIZE, N_SUBSEQUENCES, SUBSEQUENCE_LENGTH, *STATE_DIMS)
    We split along the N_SUBSEQUENCES dimension to get batches of single subsequences that can be fed into the model
    chronologically to adhere to statefulness if the following line throws a CPU to GPU error this is most likely due
    to too little memory on the GPU; lower the number of worker/horizon/... ."""
    n_trajectories_per_batch, n_chunks_per_trajectory_per_batch = batch_size

    recurrent_layers = [layer for layer in joint.submodules if isinstance(layer, tf.keras.layers.RNN)]

    batch_ent = tf.constant(0.)
    batch_pi_loss = tf.constant(0.)
    batch_v_loss = tf.constant(0.)

    batch_ents = []
    batch_pi_losses = []
    batch_v_losses = []

    split_batch = _split_batch(super_batch, n_chunks_per_trajectory_per_batch)
    for batch_i in tf.range(split_batch["advantage"].shape[1]):
        batch_grad, batch_ent, batch_pi_loss, batch_v_loss = _do_batch_op(
            joint,
            recurrent_layers,
            n_chunks_per_trajectory_per_batch,
            batch_i,
            split_batch,
            distribution,
            continuous_control,
            clip_values,
            gradient_clipping,
            clip,
            c_value,
            c_entropy,
            is_recurrent
        )
        optimizer.apply_gradients(zip(batch_grad, joint.trainable_variables))
        batch_ents.append(batch_ent)
        batch_pi_losses.append(batch_pi_loss)
        batch_v_losses.append(batch_v_loss)

        if pbar is not None:
            pbar.update(1)

    return tf.reduce_mean(batch_ents), tf.reduce_mean(batch_pi_losses), tf.reduce_mean(batch_v_losses)
