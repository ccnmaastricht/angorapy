import json
import os

import numpy as np
from matplotlib import pyplot as plt

from angorapy.common.const import PATH_TO_EXPERIMENTS, QUALITATIVE_COLOR_PALETTE

experiment_ids = ['1674074113967956', '1674074113731734', '1673350499432390']
names = ["1", "2", "3"]
reward_developments = {}
reward_bands = {}

cosucc_developments = {}
cosucc_bands = {}

for i, id in enumerate(experiment_ids):
    with open(os.path.join("../../../", PATH_TO_EXPERIMENTS, str(id), "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join("../../../", PATH_TO_EXPERIMENTS, str(id), "progress.json")) as f:
        progress = json.load(f)
    with open(os.path.join("../../../", PATH_TO_EXPERIMENTS, str(id), "statistics.json")) as f:
        statistics = json.load(f)

    exp_name = names[i]
    reward_developments[exp_name] = progress["rewards"]["mean"]
    reward_bands[exp_name] = (
        np.array(progress["rewards"]["mean"]) - .2 * np.array(progress["rewards"]["stdev"]),
        np.array(progress["rewards"]["mean"]) + .2 * np.array(progress["rewards"]["stdev"])
    )

    cosucc_developments[exp_name] = statistics["auxiliary_performances"]["consecutive_goals_reached"]["mean"]
    cosucc_bands[exp_name] = (
        np.array(statistics["auxiliary_performances"]["consecutive_goals_reached"]["mean"])
        - .2 * np.array(statistics["auxiliary_performances"]["consecutive_goals_reached"]["std"]),

        np.array(statistics["auxiliary_performances"]["consecutive_goals_reached"]["mean"])
        + .2 * np.array(statistics["auxiliary_performances"]["consecutive_goals_reached"]["std"])
    )

fig, axes = plt.subplots(1, 2)

x_max = len(list(reward_developments.items())[0][1])
for i, (name, rewards) in enumerate(reward_developments.items()):
    x_max = min(x_max, len(rewards))
    axes[0].plot(rewards, label=name, color=QUALITATIVE_COLOR_PALETTE[i])
    axes[0].fill_between(range(len(rewards)), reward_bands[name][0], reward_bands[name][1], alpha=0.1)

    axes[1].plot(cosucc_developments[name], label=name, color=QUALITATIVE_COLOR_PALETTE[i])
    axes[1].fill_between(range(len(rewards)), cosucc_bands[name][0], cosucc_bands[name][1], alpha=0.1)

axes[0].set_xlabel("Cycle")
axes[0].set_ylabel("Avg. Episode Return")
axes[0].legend()

axes[1].set_xlabel("Cycle")
axes[1].set_ylabel("Avg. Consecutive Goals Reached")
axes[1].legend()

axes[0].set_xlim(0, x_max)
axes[0].set_ylim(0)

axes[1].set_xlim(0, x_max)
axes[1].set_ylim(0)

plt.gcf().set_size_inches(16, 4)
plt.show()
# plt.savefig("../../../docs/figures/manipulate-shared-ablation.pdf", format="pdf", bbox_inches="tight")