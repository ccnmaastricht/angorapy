import multiprocessing
import os
import time

import gym
import tensorflow as tf
from tensorflow.keras.layers import Dense

from agent.policy import act_discrete
from policy_networks.fully_connected import PPOActorNetwork, PPOCriticNetwork
from utilities.util import flat_print

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def foo(model_name):
    env = gym.make("CartPole-v1")
    newmodel = tf.keras.models.load_model(model_name)
    print("PASSING")
    return act_discrete(newmodel, env.reset().reshape([1, -1]))


if __name__ == "__main__":
    # pool = multiprocessing.Pool(multiprocessing.cpu_count())
    worker_pool = multiprocessing.Pool(multiprocessing.cpu_count())

    env = gym.make("CartPole-v1")

    policy = PPOActorNetwork(env)
    # critic = PPOCriticNetwork(env)

    predef = env.reset().reshape([1, -1])
    policy.predict(predef)
    # critic.predict(predef)

    print(f"Parallelize Over {multiprocessing.cpu_count()} Threads.\n")
    for iteration in range(10):
        iteration_start = time.time()

        # run simulations in parallel
        flat_print("Gathering...")

        # export the current state of the policy and value network under unique (-enough) key
        name_key = round(time.time())
        policy.save(f"saved_models/{name_key}/policy")
        # critic.save(f"saved_models/{name_key}/value")

        results = [worker_pool.apply(foo, args=(f"saved_models/{name_key}/policy",)) for _ in range(4)]
