"""Visualize benchmark results in a model-over-task matrix of plots."""

import json
import numpy as np
import matplotlib.pyplot as plt

with open("benchmark_results.json", "r") as f:
    results = json.load(f)

fig, axs = plt.subplots(len(results), len(results["LunarLander-v2"]), figsize=(20, 20))

for i, task_name in enumerate(results):
    for j, model_name in enumerate(results[task_name]):
        for k, model_type in enumerate(results[task_name][model_name]):
            axs[i, j].plot(np.mean(results[task_name][model_name][model_type], axis=0), label=model_type)

        # set titles as row and column labels (like table)
        if j == 0:
            axs[i, j].set_ylabel(task_name)
        if i == 0:
            axs[i, j].set_title(model_name)

# shared legend
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center')

plt.show()