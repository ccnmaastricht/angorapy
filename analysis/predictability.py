#!/usr/bin/env python
"""Evaluate a loaded agent on a task."""
import argparse
import os

from analysis.investigation import Investigator
from utilities.model_utils import is_recurrent_model
from utilities.util import flatten

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from agent.ppo_agent import PPOAgent
import tensorflow as tf

tf.get_logger().setLevel('INFO')

parser = argparse.ArgumentParser(description="Inspect an episode of an agent.")
parser.add_argument("id", type=int, nargs="?", help="id of the agent, defaults to newest", default=1639582562)
parser.add_argument("--state", type=str, help="state, either iteration or 'best'", default="best")

args = parser.parse_args()
args.state = int(args.state) if args.state not in ["b", "best", "last"] else args.state

agent = PPOAgent.from_agent_state(args.id, args.state, path_modifier="../")
investigator = Investigator.from_agent(agent)
env = agent.env

is_recurrent = is_recurrent_model(investigator.network)
investigator.network.reset_states()

done, step = False, 0
state = env.reset()
prepared_state = state.with_leading_dims(time=is_recurrent).dict()
investigator.network(prepared_state, training=False)

investigator.network.summary()

print(investigator.list_layer_names())
for layer_name in investigator.list_layer_names():
    print(layer_name)
    # if isinstance(investigator.get_layer_by_name(layer_name), tf.keras.layers.TimeDistributed):
    #     continue
    activities = investigator.get_layer_activations(layer_name, prepared_state)
    # print(f"{layer_name}: {[w.shape for w in activities]}")
