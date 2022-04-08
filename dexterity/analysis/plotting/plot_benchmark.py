import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

from common.const import QUALITATIVE_COLOR_PALETTE
from utilities.plotting import plot_with_confidence

parser = argparse.ArgumentParser()
parser.add_argument("exp", nargs="*", type=str, default=["default_Acrobot-v1"],
                    help="name of the experiment")
parser.add_argument("--show", action="store_true")
parser.add_argument("--ignore", nargs="*", type=str, default=[], help="list of ignored configs")
parser.add_argument("--only", nargs="*", type=str, default=[], help="list of included configs")
parser.add_argument("--no-confidence", action="store_true")
parser.add_argument("--cycles", type=int)

args = parser.parse_args()

benchmark_data = {}
for exp in args.exp:
    with open(f"docs/benchmarks/{exp}.json") as f:
        benchmark_data.update(json.load(f))

results = benchmark_data["results"]
meta = benchmark_data["meta"]

plt.axhline(meta["reward_threshold"], ls="--", color="grey") if meta["reward_threshold"] is not None else None

x = list(range(1, len(results[list(results.keys())[0]]["means"]) + 1))

include_configs = results.keys()
if len(args.only) > 0:
    include_configs = [c for c in include_configs if c in args.only]
elif len(args.ignore) > 0:
    include_configs = [c for c in include_configs if c not in args.ignore]

for i, name in enumerate(include_configs):

    sub_exp = results[name]

    means = sub_exp["means"]
    stdevs = np.sqrt(sub_exp["var"])
    n = sub_exp["n"]
    h = st.t.interval(0.95, sub_exp["n"], loc=means, scale=stdevs / np.sqrt(n))

    plot_with_confidence(x=means[:args.cycles],
                         lb=h[0][:args.cycles],
                         ub=h[1][:args.cycles],
                         label=f"{name} [{sub_exp['n']}]",
                         col=QUALITATIVE_COLOR_PALETTE[i],
                         alpha=0 if args.no_confidence else 0.2)

    ax = plt.gca()
    # ax.set_facecolor("#FBFBFB")

    plt.xlabel("Cycle")
    plt.ylabel("Mean Cumulative Reward")

plt.legend(loc="lower right")
plt.title(f"{args.exp[0].split('_')[1]}")
plt.savefig(f"docs/benchmarks/benchmarking-{'-'.join(include_configs)}-{'-'.join(args.exp)}.pdf", format="pdf")
if args.show:
    plt.show()
