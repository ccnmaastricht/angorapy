import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time

import tensorflow as tf

from common.const import VISION_WH
from common.policies import BetaPolicyDistribution
from common.wrappers import make_env
from models import get_model_builder

# tf.config.run_functions_eagerly(False)

sequence_length = 8
batch_size = 128

model_builder = get_model_builder(model="shadow", model_type="gru", shared=False, blind=False)

env = make_env("ReachAbsoluteVisual-v0")
distribution = BetaPolicyDistribution(env)
_, _, model = model_builder(
    env, distribution,
    bs=batch_size,
    sequence_length=sequence_length
)

optimizer = tf.keras.optimizers.SGD()


def _get_data():
    sample_batch = {
        "vision": tf.random.normal([16, sequence_length, VISION_WH, VISION_WH, 3]),
        "proprioception": tf.random.normal([16, sequence_length, 48]),
        "somatosensation": tf.random.normal([16, sequence_length, 92]),
        "goal": tf.random.normal([16, sequence_length, 15])
    }

    dataset = tf.data.Dataset.from_tensor_slices(sample_batch)
    return dataset.repeat(128).batch(batch_size)


@tf.function
def _get_grads(batch):
    with tf.GradientTape() as tape:
        out, v = model(batch, training=True)
        loss = tf.reduce_mean(out - v)

    grads = tape.gradient(loss, model.trainable_variables)

    return grads


def _train():
    start_time = time.time()

    for epoch in range(100):
        dataset = _get_data()
        for batch in dataset:
            grads = _get_grads(batch)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        print(f"Finished Epoch {epoch}.")

    print(f"Execution Time: {time.time() - start_time}")


_train()
