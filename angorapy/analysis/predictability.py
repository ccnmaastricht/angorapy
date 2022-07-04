#!/usr/bin/env python
import argparse

from angorapy.agent.ppo_agent import PPOAgent
from angorapy.analysis import investigators

parser = argparse.ArgumentParser(description="Inspect an episode of an agent.")
parser.add_argument("id", type=int, nargs="?", help="id of the agent, defaults to newest", default=1653053413)
parser.add_argument("--state", type=str, help="state, either iteration or 'best'", default="best")

args = parser.parse_args()
args.state = int(args.state) if args.state not in ["b", "best", "last"] else args.state

agent = PPOAgent.from_agent_state(args.id, args.state, path_modifier="../../")
investigator = investigators.Predictability.from_agent(agent)
env = agent.env

investigator.prepare(env,
                     layers=[
                         "SSC_activation_1",
                         "SSC_activation_2",
                         "LPFC_activation",
                         "MCC_activation",
                         "IPL_activation",
                         "SPL_activation",
                         "IPS_activation",
                         "PMC_activation",
                         "M1_activation",
                     ],
                     n_states=1000)


for information in ["proprioception", "vision", "somatosensation"]:
    print(f"Predicting {information} from activity.\n-----------------------")

    investigator.measure("noise", information)
    investigator.measure("SSC_activation_1", information)
    investigator.measure("SSC_activation_2", information)
    investigator.measure("LPFC_activation", information)
    investigator.measure("MCC_activation", information)
    investigator.measure("IPL_activation", information)
    investigator.measure("SPL_activation", information)
    investigator.measure("IPS_activation", information)
    investigator.measure("PMC_activation", information)
    investigator.measure("M1_activation", information)

    print("\n\n")
