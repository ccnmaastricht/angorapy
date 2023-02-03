import itertools
import json
import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from angorapy.common.const import PATH_TO_EXPERIMENTS, QUALITATIVE_COLOR_PALETTE

experiment_ids = [['1674975602294059', '1674975602446141', '1671749718306373'],
                  ['1674343261520188', '1674308148646663', '1674074113967956', '1674074113731734', '1673350499432390']]
names = ["symmetric", "asymmetric"]
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
    cosucc_bands[exp_name] = []
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

        window_size_one_sided = 5
        cosucc_traj = statistics["auxiliary_performances"]["consecutive_goals_reached"]["mean"]
        cosucc_developments[exp_name].append(np.array([
            np.mean(np.array(cosucc_traj)[max(i - window_size_one_sided, 0):min(i + window_size_one_sided, len(cosucc_traj))]) for i in
            range(len(cosucc_traj))
        ]))
        cosucc_bands[exp_name].append(np.array([
            np.std(np.array(cosucc_traj)[
                    max(i - window_size_one_sided, 0):min(i + window_size_one_sided, len(cosucc_traj))]) for i in
            range(len(cosucc_traj))
        ]))

    try:
        evaluation_reward_histograms[exp_name] = np.mean([c for c, b in evaluation_reward_histograms[exp_name]],
                                                         axis=0), evaluation_reward_histograms[exp_name][0][1]
    except:
        pass
    # reward_developments[exp_name] = np.array(
    #     [trace[:min(map(len, reward_developments[exp_name]))] for trace in reward_developments[exp_name]])
    # cosucc_developments[exp_name] = np.array(
    #     [trace[:min(map(len, cosucc_developments[exp_name]))] for trace in cosucc_developments[exp_name]])

ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=2)
ax2 = plt.subplot2grid((1, 3), (0, 2), colspan=1)

for i, (name, rewards) in enumerate(cosucc_developments.items()):
    for j in range(len(cosucc_developments[name])):
        ax1.plot(cosucc_developments[name][j], label=name if j == 0 else None, color=QUALITATIVE_COLOR_PALETTE[i])
        ax1.fill_between(range(len(cosucc_bands[name][j])),
                         cosucc_developments[name][j] - cosucc_bands[name][j],
                         cosucc_developments[name][j] + cosucc_bands[name][j], alpha=0.2, color=QUALITATIVE_COLOR_PALETTE[i])

    # ax2.hist(
    #     evaluation_reward_histograms[name][1][:-1],
    #     evaluation_reward_histograms[name][1],
    #     weights=evaluation_reward_histograms[name][0],
    #     edgecolor="white", linewidth=0.5, alpha=0.8
    # )
    # ax2.set_xlabel("Consecutive Goals Reached")
    # ax2.set_ylabel("Number of Episodes")

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
ax1.legend(loc="upper left")

ax1.set_xlim(0)

plt.gcf().set_size_inches(12, 4)
plt.subplots_adjust(wspace=0.3, right=0.995, left=0.05, top=0.99)
plt.show()
# plt.savefig("../../../docs/figures/manipulate-asymmetry-ablation.pdf", format="pdf", bbox="tight")
