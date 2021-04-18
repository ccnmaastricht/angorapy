import json
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from common.const import QUALITATIVE_COLOR_PALETTE

AGENT_ID =  1581809147

with open(f"../../monitor/static/experiments/{AGENT_ID}/progress.json", "r") as f:
    data = json.load(f)

vloss = data["vloss"]
ploss = data["ploss"]
entropy = data["entropies"]
rewards = data["rewards"]["mean"]

fig: Figure
fig, axs = plt.subplots(1, 3)
fig.set_size_inches(16, 4)

# axs[0].set_title("Policy Gradient Loss")
# axs[1].set_title("Value Function Loss")
# axs[2].set_title("Entropy Bonus")

# plot rewards
for ax in axs:
    ax.set_zorder(10)
    ax.patch.set_visible(False)
    twin_ax = ax.twinx()
    pr = twin_ax.plot(rewards, color="lightgrey", zorder=-1)
    twin_ax.set_ylabel("Mean Cumulative Reward")

p1 = axs[0].plot(ploss, color=QUALITATIVE_COLOR_PALETTE[0], zorder=10)
p2 = axs[1].plot(vloss, color=QUALITATIVE_COLOR_PALETTE[1], zorder=10)
p3 = axs[2].plot(entropy, color=QUALITATIVE_COLOR_PALETTE[2], zorder=10)

axs[0].set_ylabel("Cycle Mean Loss")
axs[1].set_ylabel("Cycle Mean Loss")
axs[2].set_ylabel("Cycle Mean Bonus")
fig.legend([p1, p2, p3, pr],
           labels=["Policy Gradient Loss", "Value Gradient Loss", "Entropy Bonus", "Reward"],
           loc="lower center", bbox_to_anchor=(0.5, 0),
           ncol=4)

plt.subplots_adjust(wspace=.5, bottom=0.2)
plt.savefig("../../docs/figures/losses.pdf", format="pdf", bbox_inches='tight')
plt.show()
