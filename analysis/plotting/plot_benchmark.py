import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

from utilities.plotting import plot_with_confidence

colors = ["red", "green", "blue", "orange", "brown"]

parser = argparse.ArgumentParser()
parser.add_argument("exp", nargs="*", type=str, default=["default_Acrobot-v1"],
                    help="name of the experiment")
parser.add_argument("--ignore", nargs="*", type=str, default=[], help="list of ignored configs")
parser.add_argument("--no-conf", action="store_true")

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
    if name in args.ignore:
        continue

    sub_exp = results[name]

    means = sub_exp["means"]
    stdevs = np.sqrt(sub_exp["var"])
    n = sub_exp["n"]
    h = st.t.interval(0.95, sub_exp["n"], loc=means, scale=stdevs / np.sqrt(n))

    plot_with_confidence(x=means,
                         lb=h[0],
                         ub=h[1],
                         label=f"{name} [{sub_exp['n']}]",
                         col=colors[i],
                         alpha=0 if args.no_conf else 0.2)

    plt.xlabel("Cycle")
    plt.ylabel("Mean Cumulative Reward")

plt.legend()
plt.title(f"{args.exp[0].split('_')[1]} ({args.exp[0].split('_')[0]})")
plt.savefig(f"docs/benchmarks/benchmarking_{'_'.join(args.exp)}.pdf", format="pdf")
plt.show()
