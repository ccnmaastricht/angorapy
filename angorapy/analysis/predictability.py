#!/usr/bin/env python
import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from mpi4py import MPI
from angorapy.agent.ppo_agent import PPOAgent
from angorapy.analysis import investigators

mpi_comm = MPI.COMM_WORLD

parser = argparse.ArgumentParser(description="Inspect an episode of an agent.")
parser.add_argument("id", type=int, nargs="?", help="id of the agent, defaults to newest", default=1679142835973298)
parser.add_argument("--state", type=str, help="state, either iteration or 'best'", default="best")
parser.add_argument("--repeats", type=int, help="number of repetitions of the process", default=10)
parser.add_argument("--n-states", type=int, help="states per repeat", default=200000)

args = parser.parse_args()

# determine number of repetitions on this worker
worker_base, worker_extra = divmod(args.repeats, mpi_comm.size)
worker_split = [worker_base + (r < worker_extra) for r in range(mpi_comm.size)]
workers_n = worker_split[mpi_comm.rank]

args.state = int(args.state) if args.state not in ["b", "best", "last"] else args.state

agent = PPOAgent.from_agent_state(args.id, args.state, path_modifier="./")
investigator = investigators.Predictability.from_agent(agent)
env = agent.env

# nx_model = nx.drawing.nx_pydot.from_pydot(model_to_dot(investigator.network, expand_nested=True, subgraph=True))
#
# model_to_dot(investigator.network, expand_nested=True).write_png("dot_sankey.png")

# nodes_to_delete = []
# for node, data in zip(nx_model.nodes(), list(nx_model.nodes.values())):
#     if data.get("label") is None or "mask" in data["label"]:
#         nodes_to_delete.append(node)

# for node_name in nodes_to_delete:
#     nx_model.add_edge(*nx_model.predecessors(node_name), *nx_model.successors(node_name))
#     nx_model.remove_node(node_name)

# make output
# nx.drawing.nx_pydot.to_pydot(nx_model).write_png("sankey.png")

targets = ["proprioception",
           # "vision",
           # "touch",
           "reward",
           "fingertip_positions",
           "goal",
           "translation",
           "translation_to_10",
           "translation_to_50",
           "object_orientation",
           # "rotation_matrix",
           # "rotation_matrix_last_10",
           # "current_rotational_axis",
           "goals_achieved_so_far",
           # "needed_thumb"
]
results = {
    "noise": {},
    "proprioception": {},
    "vision": {},
    "touch": {},
    "goal": {},
    "SSC_activation_1": {},
    "SSC_activation_2": {},
    "LPFC_activation": {},
    "MCC_activation": {},
    "IPL_activation": {},
    "SPL_activation": {},
    "IPS_activation": {},
    "M1_activation": {},
    # "lstm_cell_1": {},
    "pmc_recurrent_layer": {},
}

merged_results = copy.deepcopy(results)

for information in targets:
    for source in results.keys():
        results[source][information] = []
        merged_results[source][information] = []

for i in range(workers_n):
    if mpi_comm.rank == 0:
        print(f"Collecting for repeat {i + 1}/{workers_n}")
    investigator.prepare(env,
                         layers=[
                             "SSC_activation_1",
                             "SSC_activation_2",
                             "LPFC_activation",
                             "MCC_activation",
                             "IPL_activation",
                             "SPL_activation",
                             "IPS_activation",
                             "pmc_recurrent_layer",
                             # "lstm_cell_1",
                             "M1_activation",
                         ], n_states=args.n_states)

    for information in targets:
        if mpi_comm.rank == 0:
            print(f"Predicting {information} from activity.")

        for source in results.keys():
            results[source][information].append(investigator.fit(source, information))

results_collection = mpi_comm.gather(results)

if mpi_comm.rank == 0:
    for result in results_collection:
        for source in result.keys():
            for information in result[source].keys():
                merged_results[source][information].extend(result[source][information])

    with open("storage/predictability_repeated_april08.json", "w") as f:
        json.dump(merged_results, f)
