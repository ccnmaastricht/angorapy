import os

import matplotlib.pyplot as plt
import seaborn as sns

from agent.ppo_agent import PPOAgent

AGENT_ID = 1582658038

os.chdir("../../../")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
agent = PPOAgent.from_agent_state(AGENT_ID, "b", path_modifier="")

# stats = StatBundle(10, 10, [3,6,3,8,6,7,4,3,6,7,4], [], None)
stats = agent.evaluate(100)

sns.set(style="whitegrid")
violin_ax = sns.violinplot(y=stats.episode_rewards)
swarm_ax = sns.swarmplot(y=stats.episode_rewards, color=".25")

plt.ylabel("Episode Reward")

plt.subplots_adjust(wspace=.5, bottom=0.2)
plt.savefig("docs/figures/reaching_evaluation.pdf", format="pdf", bbox_inches='tight')
plt.show()
