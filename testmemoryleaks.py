import os

import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time

import tensorflow as tf

from agent.dataio import read_dataset_from_storage
from common.const import VISION_WH
from common.policies import BetaPolicyDistribution
from common.wrappers import make_env
from models import get_model_builder

# tf.config.run_functions_eagerly(False)

sequence_length = 2
batch_size = 2

model_builder = get_model_builder(model="shadow", model_type="rnn", shared=False, blind=False)

env = make_env("ReachAbsoluteVisual-v0")
distribution = BetaPolicyDistribution(env)
_, _, model = model_builder(
    env, distribution,
    bs=batch_size,
    sequence_length=sequence_length
)

optimizer = tf.keras.optimizers.SGD()


def _get_data():
    dataset = read_dataset_from_storage(dtype_actions=tf.float32,
                                        id_prefix=1623091094,
                                        worker_ids=[0],
                                        responsive_senses=["proprioception", "vision", "somatosensation", "goal"])
    return dataset.repeat(128)


@tf.function
def _get_grads(batch):
    with tf.GradientTape() as tape:
        out, v = model(batch, training=True)
        loss = tf.reduce_mean(out - v)

    grads = tape.gradient(loss, model.trainable_variables)

    return grads


def _train():
    start_time = time.time()

    for cycle in range(100):
        dataset = _get_data()

        for epoch in range(3):
            batched_dataset = dataset.batch(batch_size)

            for batch in tqdm.tqdm(batched_dataset, desc=f"Cylce {cycle} Epoch {epoch}"):
                grads = _get_grads(batch)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            print(f"Finished Epoch {epoch}.")

    print(f"Execution Time: {time.time() - start_time}")


_train()
