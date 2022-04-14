#!/usr/bin/env python
import argparse

from dexterity.agent.ppo_agent import PPOAgent
from dexterity.analysis import investigators

parser = argparse.ArgumentParser(description="Inspect an episode of an agent.")
parser.add_argument("id", type=int, nargs="?", help="id of the agent, defaults to newest", default=1639582562)
parser.add_argument("--state", type=str, help="state, either iteration or 'best'", default="best")

args = parser.parse_args()
args.state = int(args.state) if args.state not in ["b", "best", "last"] else args.state

agent = PPOAgent.from_agent_state(args.id, args.state, path_modifier="../../")
investigator = investigators.Predictability.from_agent(agent)
env = agent.env

investigator.prepare(env, layers=["LPFC_activation", "IPS_activation", "PMC_activation"])
investigator.measure_predictability("noise", "proprioception")
investigator.measure_predictability("LPFC_activation", "proprioception")
investigator.measure_predictability("IPS_activation", "proprioception")
investigator.measure_predictability("PMC_activation", "proprioception")
