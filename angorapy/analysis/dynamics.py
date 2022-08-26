#!/usr/bin/env python
import argparse

from keras.utils import plot_model, model_to_dot

from angorapy.agent.ppo_agent import PPOAgent
from angorapy.analysis import investigators
import networkx as nx


parser = argparse.ArgumentParser(description="Inspect an episode of an agent.")
parser.add_argument("id", type=int, nargs="?", help="id of the agent, defaults to newest", default=1653053413)
parser.add_argument("--state", type=str, help="state, either iteration or 'best'", default="best")

args = parser.parse_args()
args.state = int(args.state) if args.state not in ["b", "best", "last"] else args.state

agent = PPOAgent.from_agent_state(args.id, args.state, path_modifier="../../")
investigator = investigators.Dynamics.from_agent(agent)
env = agent.env

investigator.prepare(env, layer="pmc_recurrent_layer", n_states=100)
investigator.fit()
