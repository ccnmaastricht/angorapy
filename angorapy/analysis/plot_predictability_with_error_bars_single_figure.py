import json
import re

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from angorapy.common.const import QUALITATIVE_COLOR_PALETTE as COLORS
import statsmodels.stats.api as sms

from angorapy.utilities.util import stack_dicts

font = {'family' : 'Times',
        'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)

partial_results = []
for filename in ["./storage/predictability_repeated_august19d.json", "./storage/predictability_repeated_august19e.json"]:
    with open(filename, "r") as f:
        partial_results.append(json.load(f))

results = {}
for key in partial_results[0].keys():
    for target in partial_results[0][key].keys():
        for part in partial_results:
            if key not in part or target not in part[key]:
                continue

            if key not in results:
                results[key] = {}
            if target not in results[key]:
                results[key][target] = []

            results[key][target] = results[key][target] + part[key][target]

# remove entries from results that are contain no underscore in their name
results = {k: v for k, v in results.items() if "_" in k or "goal" in k}

x = results.keys()

# create x labels by removing 'activation' potentially surrounded by underscores from the x label name
x_labels = [re.sub(r"(_)?internal(_)?", "", s) for s in results.keys()]

# rename x label "pmc_recurrent_layer" to "PMC"
x_labels = [re.sub(r"pmc_recurrent_layer", "PMC", s) for s in x_labels]
x_labels = [s.upper() if s != "goal" else "Goal" for s in x_labels]

targets = ["reward", "goals_achieved_so_far"]
target_names = ["Reward", "Goals Achieved So Far"]

# create a figure with a single plot
fig, axs = plt.subplots(1, len(targets), sharey="row", figsize=(8, 2))

# plot barplots for all targets, with bars placed next to each other not on top of each other
for i, target in enumerate(targets):
    # plot barplot for target
    axs[i].bar(np.arange(len(x)),
            [np.mean(results[s][target]) for s in results.keys()],
            # use confidence interval as error bars
            yerr=[sms.DescrStatsW(results[s][target]).tconfint_mean()[1] - np.mean(results[s][target]) for s in results.keys()],
            capsize=2,
            color=COLORS[i])

    # set labels of the x axis to keys in results
    axs[i].set_xticks(np.arange(len(x)))
    axs[i].set_xticklabels(x_labels)

    # make x axis labels vertical
    axs[i].tick_params(axis='x', rotation=90)

    if i == 0:
        axs[i].set_ylabel("Predictability (R2)")
    else:
        axs[i].tick_params(axis='y', which='both', left=False, labelleft=False)

    axs[i].set_ylim(0, 1)
    axs[i].set_title(target_names[i])

# remove space between subplots
plt.subplots_adjust(wspace=0.05)

plt.savefig(f"predictability_reward_goals.pdf", format="pdf", bbox_inches="tight")