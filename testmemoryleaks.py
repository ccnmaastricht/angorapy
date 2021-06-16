import os

import psutil
import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time

import tensorflow as tf

from agent.dataio import read_dataset_from_storage
from agent.ppo.optim import learn_on_batch
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
                                        responsive_senses=["proprioception", "vision", "somatosensation", "goal"])
    return dataset


def _train():
    start_time = time.time()

    for cycle in range(100):
        dataset = _get_data()

        for epoch in range(3):
            batched_dataset = dataset.batch(batch_size)

            for b in batched_dataset:
                print(b)
                split_batch = {bk: tf.split(bv, bv.shape[1], axis=1) for bk, bv in b.items()}
                for i in range(len(split_batch["advantage"])):
                    # extract subsequence and squeeze away the N_SUBSEQUENCES dimension
                    partial_batch = {k: tf.squeeze(v[i], axis=1) for k, v in split_batch.items()}
                    grads = learn_on_batch(partial_batch, model, distribution, True, True, tf.constant(0.2), None,
                                           tf.constant(1), tf.constant(0.1), True)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

        print(f"Finished Cycle {cycle} with {round(psutil.virtual_memory()[3] / 1e9, 2)}.")

    print(f"Execution Time: {time.time() - start_time}")


_train()
