import unittest

from angorapy import make_task
from angorapy.agent.ppo_agent import PPOAgent
from angorapy.models import get_model_builder


class TestPostProcessors(unittest.TestCase):

    def test_postprocessors(self):
        model_builder = get_model_builder("simple", "ffn")
        env = make_task("LunarLanderContinuous-v2")

        agent = PPOAgent(model_builder, env, )
        agent.drill(5)
        agent.save_agent_state("test")

        new_agent = PPOAgent.from_agent_state(agent.agent_id, from_iteration="test")
        new_agent.drill(5)

        print(new_agent.wrapper_stat_history)