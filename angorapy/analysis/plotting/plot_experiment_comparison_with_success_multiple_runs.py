import itertools
import json
import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from angorapy.common.const import PATH_TO_EXPERIMENTS, QUALITATIVE_COLOR_PALETTE

# experiment_ids = [['1674343261520188', '1674308148646663', '1674074113967956', '1674074113731734', '1673350499432390']]  # only best setting
# experiment_ids = [['1674343261520188', '1674308148646663', '1674074113967956', '1673350499432390']]  # only best setting
# experiment_ids = [['1674074113967956', '1673350499432390', '1674074113731734']]  # only best setting
# names = ["asymmetric"]

# experiment_ids = [['1673786170549564'], ['1673350499432390']]  # shared vs unshared
# experiment_ids = [['1674975602294059', '1674975602446141', '1671749718306373'],
#                   ['1674343261520188', '1674308148646663', '1674074113967956', '1674074113731734', '1673350499432390']]  # symmetric vs asymmetric
# names = ["symmetric", "asymmetric"]

# experiment_ids = [['1673350499432390']]
# names = ["asymmetric"]

experiment_ids = [['1675028736765791', '1674983643322591'],
                  ['1674985163288377', '1674983177286330'],
                  ['1674343261520188', '1674308148646663', '1674074113967956', '1673350499432390']]  # compare distributions
names = ["beta", "gaussian", "multicategorical"]
# names = ["beta", "multicategorical"]
reward_developments = {}
reward_bands = {}

cosucc_developments = {}
cosucc_bands = {}

evaluation_rewards = {}
evaluation_reward_histograms = {}

for i, group in enumerate(experiment_ids):
    exp_name = names[i]
    reward_developments[exp_name] = []
    cosucc_developments[exp_name] = []
    evaluation_rewards[exp_name] = []
    evaluation_reward_histograms[exp_name] = []

    for id in group:
        with open(os.path.join("../../../", PATH_TO_EXPERIMENTS, str(id), "meta.json")) as f:
            meta = json.load(f)
        with open(os.path.join("../../../", PATH_TO_EXPERIMENTS, str(id), "progress.json")) as f:
            progress = json.load(f)
        with open(os.path.join("../../../", PATH_TO_EXPERIMENTS, str(id), "statistics.json")) as f:
            statistics = json.load(f)

        try:
            with open(os.path.join("../../../", PATH_TO_EXPERIMENTS, str(id), "evaluation.json")) as f:
                evaluation = json.load(f)

            evaluation_rewards[exp_name] += evaluation["auxiliary_performances"]["consecutive_goals_reached"]
            evaluation_reward_histograms[exp_name].append(
                np.histogram(evaluation["auxiliary_performances"]["consecutive_goals_reached"], bins=50))
        except:
            pass

        reward_developments[exp_name].append(progress["rewards"]["mean"])
        cosucc_developments[exp_name].append(statistics["auxiliary_performances"]["consecutive_goals_reached"]["mean"])

    try:
        evaluation_reward_histograms[exp_name] = np.mean([c for c, b in evaluation_reward_histograms[exp_name]], axis=0), \
            evaluation_reward_histograms[exp_name][0][1]
    except:
        pass
    reward_developments[exp_name] = np.array(
        [trace[:min(map(len, reward_developments[exp_name]))] for trace in reward_developments[exp_name]])
    cosucc_developments[exp_name] = np.array(
        [trace[:min(map(len, cosucc_developments[exp_name]))] for trace in cosucc_developments[exp_name]])

    rd_mean = np.mean(reward_developments[exp_name], axis=0)
    cd_mean = np.mean(cosucc_developments[exp_name], axis=0)
    rd_std = np.std(reward_developments[exp_name], axis=0)
    cd_std = np.std(cosucc_developments[exp_name], axis=0)
    reward_bands[exp_name] = (rd_mean - rd_std, rd_mean + rd_std)
    cosucc_bands[exp_name] = (cd_mean - cd_std, cd_mean + cd_std)

if len(names) <= 2:
    axes = [
        plt.subplot2grid((1, 5), (0, 0), colspan=2),
        plt.subplot2grid((1, 5), (0, 2), colspan=2),
        plt.subplot2grid((1, 5), (0, 4), colspan=1),
    ]
else:
    axes = [
        plt.subplot2grid((1, 3), (0, 0), colspan=1),
        plt.subplot2grid((1, 3), (0, 1), colspan=1),
        plt.subplot2grid((1, 3), (0, 2), colspan=1),
    ]

x_max = len(list(reward_developments.items())[0][1][0])
y_max_x0 = -np.inf
for i, (name, rewards) in enumerate(reward_developments.items()):
    x_max = max(x_max, len(rewards[0]))
    axes[0].plot(np.mean(rewards, axis=0), label=name, color=QUALITATIVE_COLOR_PALETTE[i])
    axes[0].fill_between(range(len(reward_bands[name][0])), reward_bands[name][0], reward_bands[name][1], alpha=0.2)
    y_max_x0 = max(y_max_x0, np.max(reward_bands[name][1]))

    axes[1].plot(np.mean(cosucc_developments[name], axis=0), label=name, color=QUALITATIVE_COLOR_PALETTE[i])
    axes[1].fill_between(range(len(cosucc_bands[name][0])), cosucc_bands[name][0], cosucc_bands[name][1], alpha=0.2)

    # axes[2].hist(
    #     evaluation_reward_histograms[name][1][:-1],
    #     evaluation_reward_histograms[name][1],
    #     weights=evaluation_reward_histograms[name][0],
    #     edgecolor="white", linewidth=0.5, alpha=0.8
    # )
    # axes[2].set_xlabel("Consecutive Goals Reached")
    # axes[2].set_ylabel("Number of Episodes")

df = pd.DataFrame(
    {"group": itertools.chain(*[[name] * len(dp) for name, dp in evaluation_rewards.items()]),
     "Consecutive Goals Reached": np.concatenate([dp for name, dp in evaluation_rewards.items()])}
)

sns.boxplot(data=df, x="group", y="Consecutive Goals Reached", medianprops={"color": "red"}, flierprops={"marker": "x"},
            fliersize=1, ax=axes[2], palette={name: QUALITATIVE_COLOR_PALETTE[i] for i, name in enumerate(names)},
            showmeans=True, meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"})
axes[2].set_xlabel("")
if len(names) == 1:
    axes[2].set_xticks([])

axes[0].set_xlabel("Cycle")
axes[0].set_ylabel("Avg. Episode Return")

if len(names) > 1:
    axes[0].legend()

axes[1].set_xlabel("Cycle")
axes[1].set_ylabel("Avg. Consecutive Goals Reached")

if len(names) > 1:
    axes[1].legend()

axes[0].set_xlim(0, x_max)
axes[0].set_ylim(top=y_max_x0 * 1.1)
axes[1].set_xlim(0, x_max)
axes[1].set_ylim(0, 50)
axes[2].set_ylim(0, 50)

plt.gcf().set_size_inches(12, 4)
plt.subplots_adjust(wspace=0.5, right=0.995, left=0.05, top=0.99)
plt.show()
# plt.savefig("../../../docs/figures/manipulate-learning-curves.pdf", format="pdf", bbox="tight")
