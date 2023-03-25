import json
import os

from matplotlib import pyplot as plt

from angorapy.common.const import PATH_TO_EXPERIMENTS, QUALITATIVE_COLOR_PALETTE

experiment_ids = ['1653053413', '1655284851', '1654708464']

reward_developments = {}
for id in experiment_ids:
    with open(os.path.join("../../../", PATH_TO_EXPERIMENTS, str(id), "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join("../../../", PATH_TO_EXPERIMENTS, str(id), "progress.json")) as f:
        progress = json.load(f)

    exp_name = meta["hyperparameters"]["distribution"]
    reward_developments[exp_name] = progress["rewards"]["mean"]

for i, (name, rewards) in enumerate(reward_developments.items()):
    plt.plot(rewards[:800], label=name, color=QUALITATIVE_COLOR_PALETTE[i])

plt.title("In-Hand Object Manipulation")
plt.xlabel("Cycle")
plt.ylabel("Avg. Episode Return")
plt.legend()

plt.xlim(0, 800)
plt.ylim(0)

plt.gcf().set_size_inches(16, 4)
plt.show()
# plt.savefig("../../../docs/figures/manipulate-progress.pdf", format="pdf", bbox_inches="tight")