import json
import os

from matplotlib import pyplot as plt

from common.const import PATH_TO_EXPERIMENTS

experiment_ids = [1626473870, 1626473895, 1626473919, 1626473914]

reward_developments = {}
for id in experiment_ids:
    with open(os.path.join("../../", PATH_TO_EXPERIMENTS, str(id), "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join("../../", PATH_TO_EXPERIMENTS, str(id), "progress.json")) as f:
        progress = json.load(f)

    exp_name = meta["hyperparameters"]["model"].upper()
    reward_developments[exp_name] = progress["rewards"]["mean"]

for name, rewards in reward_developments.items():
    plt.plot(rewards, label=name)

plt.xlabel("Epoch")
plt.ylabel("Reward")
plt.legend()

plt.gcf().set_size_inches(8, 4)
# plt.show()
plt.savefig("model-comparison-reach.pdf", format="pdf", bbox_inches="tight")