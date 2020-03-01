import json
import os
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.signal import savgol_filter

import seaborn as sns

from agent.ppo import PPOAgent
from utilities.const import QUALITATIVE_COLOR_PALETTE
from utilities.datatypes import StatBundle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

AGENT_ID = 1582658038
os.chdir("../../")

with open(f"monitor/static/experiments/{AGENT_ID}/progress.json", "r") as f:
    data = json.load(f)

with open(f"monitor/static/experiments/{AGENT_ID}/meta.json", "r") as f:
    meta = json.load(f)

agent = PPOAgent.from_agent_state(AGENT_ID, "b", path_modifier="")

mean_rewards = data["rewards"]["mean"]
mean_rewards_smooth = savgol_filter(mean_rewards, 51, 3)
std_rewards = data["rewards"]["stdev"]

axs: List[Axes]
fig: Figure = plt.figure(figsize=(12, 4))
grid = plt.GridSpec(1, 3)

progression_ax = fig.add_subplot(grid[:2])
progression_ax.set_xlim(0, len(mean_rewards))

progression_ax.axhline(meta["environment"]["reward_threshold"], ls="--", color="grey") \
    if meta["environment"]["reward_threshold"] is not None else None
progression_ax.plot(mean_rewards, color=QUALITATIVE_COLOR_PALETTE[0], alpha=.4, label="True Per Cycle Rewards")
progression_ax.plot(mean_rewards_smooth, color=QUALITATIVE_COLOR_PALETTE[0], label="Smoothed Per Cycle Rewards", lw=1)
progression_ax.legend(loc="lower right")

progression_ax.set_ylabel("Mean Episode Reward")
progression_ax.set_xlabel("Cycle")

# evaluate
stats = agent.evaluate(100)

boxplot_ax = fig.add_subplot(grid[2])

box_ax = sns.boxplot(y=stats.episode_rewards, ax=boxplot_ax, color=QUALITATIVE_COLOR_PALETTE[0], boxprops=dict(alpha=.4), )
sns.swarmplot(y=stats.episode_rewards, color=QUALITATIVE_COLOR_PALETTE[0], ax=boxplot_ax)

box_ax.set(xlabel="Best Model Evaluation")

plt.ylabel("Episode Reward")

plt.subplots_adjust(wspace=.5, bottom=0.2)
plt.savefig("docs/figures/reaching_results.pdf", format="pdf", bbox_inches='tight')
plt.show()
