from angorapy.agent.ppo_agent import PPOAgent
from angorapy.analysis.investigation import Investigator

agent_id = 1652356522

agent = PPOAgent.from_agent_state(agent_id, "best")
print(f"Agent {agent.agent_id} successfully loaded.")

investigator = Investigator.from_agent(agent)

# render 100 episodes
for i in range(100):
    investigator.render_episode(agent.env, act_confidently=True)
