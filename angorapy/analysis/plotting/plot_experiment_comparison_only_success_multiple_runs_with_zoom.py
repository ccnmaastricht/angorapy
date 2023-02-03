import itertools
import json
import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from angorapy.common.const import PATH_TO_EXPERIMENTS, QUALITATIVE_COLOR_PALETTE
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


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
        with open(os.path.join("../../../", PATH_TO_EXPERIMENTS, str(id), "evaluation.json")) as f:
            evaluation = json.load(f)

        reward_developments[exp_name].append(progress["rewards"]["mean"])
        cosucc_developments[exp_name].append(statistics["auxiliary_performances"]["consecutive_goals_reached"]["mean"])
        evaluation_rewards[exp_name] += evaluation["auxiliary_performances"]["consecutive_goals_reached"]
        evaluation_reward_histograms[exp_name].append(
            np.histogram(evaluation["auxiliary_performances"]["consecutive_goals_reached"], bins=50))

    evaluation_reward_histograms[exp_name] = np.mean([c for c, b in evaluation_reward_histograms[exp_name]], axis=0), \
    evaluation_reward_histograms[exp_name][0][1]
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

ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=2)
ax2 = plt.subplot2grid((1, 3), (0, 2), colspan=1)
axins = zoomed_inset_axes(ax1, 2, loc=1) # zoom = 2

x_max = len(list(reward_developments.items())[0][1][0])
for i, (name, rewards) in enumerate(reward_developments.items()):
    x_max = max(x_max, len(rewards[0]))
    ax1.plot(np.mean(cosucc_developments[name], axis=0), label=name, color=QUALITATIVE_COLOR_PALETTE[i])
    ax1.fill_between(range(len(cosucc_bands[name][0])), cosucc_bands[name][0], cosucc_bands[name][1], alpha=0.2)

    axins.plot(np.mean(cosucc_developments[name], axis=0), label=name, color=QUALITATIVE_COLOR_PALETTE[i])
    axins.fill_between(range(len(cosucc_bands[name][0])), cosucc_bands[name][0], cosucc_bands[name][1], alpha=0.2)

    # ax2.hist(
    #     evaluation_reward_histograms[name][1][:-1],
    #     evaluation_reward_histograms[name][1],
    #     weights=evaluation_reward_histograms[name][0],
    #     edgecolor="white", linewidth=0.5, alpha=0.8
    # )
    # ax2.set_xlabel("Consecutive Goals Reached")
    # ax2.set_ylabel("Number of Episodes")
axins.set_xlim(0, 100.)
axins.set_ylim(-5., 10.)
plt.xticks(visible=False)
plt.yticks(visible=False)
mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5")
# plt.draw()
df = pd.DataFrame(
    {"group": itertools.chain(*[[name] * len(dp) for name, dp in evaluation_rewards.items()]),
     "Consecutive Goals Reached": np.concatenate([dp for name, dp in evaluation_rewards.items()])}
)

sns.boxplot(data=df, x="group", y="Consecutive Goals Reached", medianprops={"color": "red"}, flierprops={"marker": "x"},
            fliersize=1, ax=ax2, palette={name: QUALITATIVE_COLOR_PALETTE[i] for i, name in enumerate(names)},
            showmeans=True, meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"})
ax2.set_xlabel("")


ax1.set_xlabel("Cycle")
ax1.set_ylabel("Avg. Consecutive Goals Reached")
ax1.legend()

ax1.set_xlim(0, x_max)
ax1.set_ylim(-5)

plt.gcf().set_size_inches(12, 4)
plt.subplots_adjust(wspace=0.3, right=0.995, left=0.05, top=0.99)
plt.show()
# plt.savefig("../../../docs/figures/manipulate-asymmetry-ablation.pdf", format="pdf", bbox="tight")
