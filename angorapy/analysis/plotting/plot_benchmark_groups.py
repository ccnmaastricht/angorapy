import json
import os
import re
from json import JSONDecodeError

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from angorapy.common.const import PATH_TO_EXPERIMENTS, BASE_SAVE_PATH

matplotlib.use('TkAgg')

group_names = ["benchmark-performance-pdl", "benchmark-performance-llc", "benchmark-performance-cp",
               "benchmark-performance-ab"]
titles = ["Pendulum", "LunarLanderContinuous", "CartPole", "Acrobot"]

# group_names = [
#     "benchmark-performance-ant",
#     "benchmark-performance-walker2d",
#     "benchmark-performance-swimmer",
#     "benchmark-performance-reacher",
#     "benchmark-performance-humanoidstandup",
#     "benchmark-performance-humanoid",
#     "benchmark-performance-hopper",
#     "benchmark-performance-halfcheetah"
# ]

# group_names = [
#     "benchmark-beta-reach",
#     "benchmark-beta-freereach"
# ]

# titles = [n.split("-")[-1].capitalize() for n in group_names]

exp_dir = "../../../" + PATH_TO_EXPERIMENTS
experiment_paths = [os.path.join(exp_dir, p) for p in os.listdir(exp_dir)]

environments = {}
reward_thresholds = {}
experiments_by_groups = {}
envs_available = set()

for exp_path in experiment_paths:

    eid_m = re.match("[0-9]+", str(exp_path.split("/")[-1]))
    if eid_m:
        eid = eid_m.group(0)
        model_path = os.path.join(BASE_SAVE_PATH, eid)

        if os.path.isfile(os.path.join(exp_path, "progress.json")):
            with open(os.path.join(exp_path, "progress.json"), "r") as f:
                progress = json.load(f)

            with open(os.path.join(exp_path, "meta.json"), "r") as f:
                try:
                    meta = json.load(f)
                except JSONDecodeError as jserr:
                    continue

            exp_group = meta.get("experiment_group", "n/a")

            if exp_group not in group_names:
                continue

            reward_threshold = None if meta["environment"]["reward_threshold"] == "None" else float(
                meta["environment"]["reward_threshold"])

            if not exp_group in experiments_by_groups.keys():
                experiments_by_groups[exp_group] = {}
                reward_thresholds[exp_group] = reward_threshold
                environments[exp_group] = meta["environment"]["name"]

            envs_available.add(meta["environment"]["name"])

            experiments_by_groups[exp_group].update({
                eid: progress
            })

n_rows, n_cols = 1, 4
fig, axs = plt.subplots(n_rows, n_cols)
fig.set_size_inches(16, 3 * n_rows)

if not isinstance(axs[0], list):
    axs = [axs]

for i, name in enumerate(group_names):
    data = experiments_by_groups[name]
    reward_trajectories = list(map(lambda x: x["rewards"]["mean"], data.values()))
    max_length = max([len(x) for x in reward_trajectories])
    padded_reward_trajectories = list(map(lambda x: np.pad(x, (0, max_length - len(x)),
                                                           mode="constant",
                                                           constant_values=np.nan), reward_trajectories))
    mean_reward = np.ma.mean(np.ma.array(padded_reward_trajectories, mask=np.isnan(padded_reward_trajectories)), axis=0)
    std_reward = np.ma.std(np.ma.array(padded_reward_trajectories, mask=np.isnan(padded_reward_trajectories)), axis=0)

    ax = axs[i // n_cols][i % n_cols]

    ax.plot(mean_reward)
    ax.fill_between(range(mean_reward.shape[0]), mean_reward - std_reward, mean_reward + std_reward, alpha=.2)

    ax.set_xlim(0, mean_reward.shape[0] - 1)
    ax.set_ylim(np.min(mean_reward - std_reward), np.max(mean_reward + std_reward))
    ax.set_xlabel("Cycle")
    ax.set_title(titles[i])

    if i % n_cols == 0:
        ax.set_ylabel("Episode Return")

plt.savefig(f"../../../docs/figures/benchmarks/{'_'.join(titles)}_benchmark.pdf", format="pdf", bbox_inches='tight')
plt.show()
