import time

import tensorflow as tf
from tqdm import tqdm

from common.const import VISION_WH
from common.policies import BetaPolicyDistribution
from common.wrappers import make_env
from models import get_model_builder

# tf.config.run_functions_eagerly(False)

sequence_length = 100
batch_size = 256

model_builder = get_model_builder(model="shadow", model_type="gru", shared=False, blind=False)

env = make_env("ReachAbsoluteVisual-v0")
distribution = BetaPolicyDistribution(env)
_, _, model = model_builder(
    env, distribution,
    bs=batch_size,
    sequence_length=sequence_length
)

optimizer = tf.keras.optimizers.SGD()

sample_batch = {
    "vision": tf.random.normal([batch_size, sequence_length, VISION_WH, VISION_WH, 3]),
    "proprioception": tf.random.normal([batch_size, sequence_length, 48]),
    "somatosensation": tf.random.normal([batch_size, sequence_length, 92]),
    "goal": tf.random.normal([batch_size, sequence_length, 15])
}

@tf.function
def _train():
    start_time = time.time()

    for _ in tqdm(range(sequence_length), disable=False):
        with tf.GradientTape() as tape:
            out, v = model(sample_batch, training=True)
            loss = tf.reduce_mean(out - v)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    print(f"Execution Time: {time.time() - start_time}")


_train()
