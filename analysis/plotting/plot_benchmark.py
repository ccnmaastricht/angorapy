import json
import math

import matplotlib.pyplot as plt
import argparse
import numpy as np

from utilities.plotting import plot_with_confidence

colors = ["red", "green", "blue", "orange"]

parser = argparse.ArgumentParser()
parser.add_argument("exp", nargs="*", type=str, default=["default_Acrobot-v1"],
                    help="name of the experiment")

args = parser.parse_args()

benchmark_data = {}
for exp in args.exp:
    with open(f"docs/benchmarks/{exp}.json") as f:
        benchmark_data.update(json.load(f))

results = benchmark_data["results"]
meta = benchmark_data["meta"]

plt.axhline(meta["reward_threshold"]) if meta["reward_threshold"] is not None else None

x = list(range(1, len(results[list(results.keys())[0]]["means"]) + 1))

for i, name in enumerate(results):
    sub_exp = results[name]

    means = sub_exp["means"]
    stdevs = np.sqrt(sub_exp["var"])

    plot_with_confidence(x=means,
                         lb=np.array(means) - np.array(stdevs),
                         ub=np.array(means) + np.array(stdevs),
                         label=name,
                         col=colors[i])

    plt.xlabel("Cycle")
    plt.ylabel("Mean Cumulative Reward")

plt.legend()
plt.title(f"{args.exp[0].split('_')[0]}")
plt.savefig(f"docs/benchmarks/benchmarking_{'_'.join(args.exp)}.pdf", format="pdf")
plt.show()
