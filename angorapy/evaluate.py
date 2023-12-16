#!/usr/bin/env python
"""Evaluate a loaded agent on a task."""

import argparse
import os
import sys

from angorapy.utilities.evaluation import evaluate_agent

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from angorapy.agent.ppo_agent import PPOAgent

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

tf.get_logger().setLevel('INFO')

parser = argparse.ArgumentParser(description="Evaluate an agent.")
parser.add_argument("id", type=int, nargs="?", help="id of the agent, defaults to newest", default="1702037093415220")
parser.add_argument("-n", type=int, help="number of evaluation episodes", default=12)
parser.add_argument("--act-confidently", action="store_true", help="act deterministically without stochasticity")
args = parser.parse_args()

agent = PPOAgent.from_agent_state(args.id, path_modifier='./../')
evaluate_agent(agent, args.n, args.act_confidently)
