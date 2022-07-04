#!/usr/bin/env python
import argparse

from angorapy.agent.ppo_agent import PPOAgent
from angorapy.analysis import investigators

parser = argparse.ArgumentParser(description="Inspect an episode of an agent.")
parser.add_argument("id", type=int, nargs="?", help="id of the agent, defaults to newest", default=1656335587945136)
parser.add_argument("--state", type=str, help="state, either iteration or 'best'", default="best")

args = parser.parse_args()
args.state = int(args.state) if args.state not in ["b", "best", "last"] else args.state

agent = PPOAgent.from_agent_state(args.id, args.state, path_modifier="../../")
investigator = investigators.Predictability.from_agent(agent)
env = agent.env

investigator.prepare(env,
                     layers=[
                         "LPFC_activation",
                         "SSC_1",
                         "SSC_2",
                         "SSC_activation_2",
                         "SSC_activation_1",
                         "M1_activation",
                         "IPS_activation",
                         "PMC_activation"],
                     n_states=1000)

investigator.measure("noise", "proprioception")
investigator.measure("SSC_1", "proprioception")
investigator.measure("SSC_2", "proprioception")
investigator.measure("LPFC_activation", "proprioception")
investigator.measure("IPS_activation", "proprioception")
investigator.measure("PMC_activation", "proprioception")
investigator.measure("M1_activation", "proprioception")
