import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

action_prob = tf.random.normal((2, 16))
old_action_prob = tf.random.normal((2, 16)).numpy()
advantage = tf.random.normal((2, 16)).numpy()
old_action_prob[0, -3:] = 0
old_action_prob[1, -5:] = 0
advantage[0, -3:] = 0
advantage[1, -5:] = 0

old_action_prob = tf.convert_to_tensor(old_action_prob)
advantage = tf.convert_to_tensor(advantage)

with tf.GradientTape() as tape:
    r = action_prob / old_action_prob
    clipped = tf.maximum(
        - tf.math.multiply(r, advantage),
        - tf.math.multiply(tf.clip_by_value(r, 1 - 0.2, 1 + 0.2), advantage)
    )

    # build and apply a mask over the probabilities (recurrent)
    mask = tf.not_equal(old_action_prob, 0)
    clipped_safe = tf.where(mask, clipped, tf.zeros_like(clipped))
    clipped_masked = tf.where(mask, clipped_safe, 0)  # masking with tf.where because inf * 0 = nan...
    out = tf.reduce_sum(clipped_masked) / tf.reduce_sum(tf.cast(mask, tf.float32))

print(tape.gradient(action_prob, out))
