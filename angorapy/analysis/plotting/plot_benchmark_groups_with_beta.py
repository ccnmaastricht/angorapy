import itertools
import json
import os
import re
from json import JSONDecodeError
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from angorapy.common.const import PATH_TO_EXPERIMENTS, BASE_SAVE_PATH, QUALITATIVE_COLOR_PALETTE

matplotlib.use('TkAgg')

group_names = {
    "gaussian": [
        "benchmark-performance-ant",
        "benchmark-performance-walker2d",
        "benchmark-performance-swimmer",
        "benchmark-gaussian-reach",
        "benchmark-performance-reacher",
        "benchmark-performance-hopper",
        "benchmark-performance-halfcheetah",
        "benchmark-gaussian-freereach"
    ], "beta": [
        "benchmark-beta-ant",
        "benchmark-beta-walker2d",
        "benchmark-beta-swimmer",
        "benchmark-beta-reach",
        "benchmark-beta-reacher",
        "benchmark-beta-hopper",
        "benchmark-beta-halfcheetah",
        "benchmark-beta-freereach"
    ]
}

titles = [n.split("-")[-1].capitalize() for n in group_names[list(group_names.keys())[0]]]

# group_names = {"any": ["benchmark-performance-pdl", "benchmark-performance-llc", "benchmark-performance-cp",
#                "benchmark-performance-ab"]}
# titles = ["Pendulum", "LunarLanderContinuous", "CartPole", "Acrobot"]

exp_dir = "../../../" + PATH_TO_EXPERIMENTS
experiment_paths = [os.path.join(exp_dir, p) for p in os.listdir(exp_dir)]

environments = {category: {} for category in group_names.keys()}
reward_thresholds = {category: {} for category in group_names.keys()}
experiments_by_groups = {category: {} for category in group_names.keys()}
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

            if exp_group not in itertools.chain(*group_names.values()):
                continue

            reward_threshold = None if meta["environment"]["reward_threshold"] == "None" else float(
                meta["environment"]["reward_threshold"])

            for category in group_names.keys():
                if exp_group in group_names[category] and exp_group not in experiments_by_groups[category].keys():
                    experiments_by_groups[category][exp_group] = {}
                    reward_thresholds[category][exp_group] = reward_threshold
                    environments[category][exp_group] = meta["environment"]["name"]

            envs_available.add(meta["environment"]["name"])

            for category in group_names.keys():
                if exp_group in group_names[category]:
                    experiments_by_groups[category][exp_group].update({
                        eid: progress
                    })

n_rows, n_cols = 2, 4
fig, axs = plt.subplots(n_rows, n_cols)
fig.set_size_inches(16, 4 * n_rows)

if not isinstance(axs[0], Iterable):
    axs = [axs]


for i_cat, category in enumerate(group_names.keys()):
    for i, name in enumerate(group_names[category]):
        data = experiments_by_groups[category][name]
        reward_trajectories = list(map(lambda x: x["rewards"]["mean"], data.values()))
        max_length = max([len(x) for x in reward_trajectories])
        padded_reward_trajectories = list(map(lambda x: np.pad(x, (0, max_length - len(x)),
                                                               mode="constant",
                                                               constant_values=np.nan), reward_trajectories))
        mean_reward = np.ma.mean(np.ma.array(padded_reward_trajectories, mask=np.isnan(padded_reward_trajectories)),
                                 axis=0)
        std_reward = np.ma.std(np.ma.array(padded_reward_trajectories, mask=np.isnan(padded_reward_trajectories)),
                               axis=0)

        ax: Axes = axs[i // n_cols][i % n_cols]

        if reward_thresholds[category][name] is not None:
            ax.axhline(reward_thresholds[category][name], color=QUALITATIVE_COLOR_PALETTE[2], ls="--")
        ax.plot(mean_reward, label=category, color=QUALITATIVE_COLOR_PALETTE[i_cat])
        ax.fill_between(range(mean_reward.shape[0]), mean_reward - std_reward, mean_reward + std_reward, alpha=.2)

        ax.set_xlim(0, mean_reward.shape[0] - 1)
        ax.set_ylim(min(np.min(mean_reward - std_reward), ax.get_ylim()[0]),
                    max(np.max(mean_reward + std_reward) * 1.1, ax.get_ylim()[1]))
        ax.set_xlabel("Cycle")
        ax.set_title(titles[i])

        if titles[i] in ["Reach", "Freereach"]:
            ax.set_title(titles[i], fontstyle="italic")

        if i % n_cols == 0:
            ax.set_ylabel("Episode Return")

        if len(group_names.keys()) > 1:
            ax.legend(loc="lower right")

plt.subplots_adjust(top=0.8, bottom=0.2, hspace=0.35, wspace=0.2)

plt.savefig(f"../../../docs/figures/benchmarks/{'_'.join(titles)}_benchmark_comparison.pdf", format="pdf",
            bbox_inches='tight')
plt.show()
