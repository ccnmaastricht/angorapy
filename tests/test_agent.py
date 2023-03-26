import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    from mpi4py import MPI

    is_root = MPI.COMM_WORLD.rank == 0
except:
    is_root = True

from angorapy.agent.ppo_agent import PPOAgent
from angorapy.common.policies import BetaPolicyDistribution, MultiCategoricalPolicyDistribution
from angorapy.common.transformers import RewardNormalizationTransformer, StateNormalizationTransformer
from angorapy.common.wrappers import make_env

from angorapy.models import get_model_builder

import unittest


def _test_drill(env_name, model_builder=None,):
    """Test drilling an agent."""
    wrappers = [StateNormalizationTransformer, RewardNormalizationTransformer]
    env = make_env(env_name, reward_config=None, transformers=wrappers)
    if model_builder is None:
        build_models = get_model_builder(model="simple", model_type="ffn", shared=False)
    else:
        build_models = model_builder
    agent = PPOAgent(build_models, env, workers=2, horizon=128)
    agent.drill(n=2, epochs=2, batch_size=64)


class AgentTest(unittest.TestCase):

    def test_drill_continuous(self):
        """Test drilling of continuous agent (LunarLanderContinuous)."""

        try:
            _test_drill("LunarLanderContinuous-v2")
        except Exception:
            self.fail("Continuous drill raises error.")

    def test_drill_discrete(self):
        """Test drilling of discrete agent (LunarLander)."""

        try:
            _test_drill("LunarLander-v2")
        except Exception:
            self.fail("Discrete drill raises error.")

    def test_drill_manipulate_multicontinuous(self):
        """Test drilling of discrete agent (LunarLander)."""

        try:
            wrappers = [StateNormalizationTransformer, RewardNormalizationTransformer]
            env = make_env("HumanoidManipulateBlockDiscreteAsynchronous-v0", reward_config=None, transformers=wrappers)
            build_models = get_model_builder(model="shadow", model_type="lstm", shared=False)
            agent = PPOAgent(build_models, env, workers=2, horizon=128, distribution=MultiCategoricalPolicyDistribution(env))
            agent.drill(n=2, epochs=2, batch_size=64)
        except Exception:
            self.fail("HumanoidManipulateBlockDiscreteAsynchronous drill raises error.")

    def test_drill_manipulate_continuous(self):
        """Test drilling of discrete agent (LunarLander)."""

        try:
            wrappers = [StateNormalizationTransformer, RewardNormalizationTransformer]
            env = make_env("HumanoidManipulateBlockAsynchronous-v0", reward_config=None, transformers=wrappers)
            build_models = get_model_builder(model="shadow", model_type="lstm", shared=False)
            agent = PPOAgent(build_models, env, workers=2, horizon=128, distribution=BetaPolicyDistribution(env))
            agent.drill(n=2, epochs=2, batch_size=64)
        except Exception:
            self.fail("HumanoidManipulateBlockDiscreteAsynchronous drill raises error.")

    def test_drill_reach(self):
        """Test drilling of discrete agent (LunarLander)."""

        try:
            wrappers = [StateNormalizationTransformer, RewardNormalizationTransformer]
            env = make_env("ReachAbsolute-v0", reward_config=None, transformers=wrappers)
            build_models = get_model_builder(model="shadow", model_type="lstm", shared=False)
            agent = PPOAgent(build_models, env, workers=2, horizon=128, distribution=BetaPolicyDistribution(env))
            agent.drill(n=2, epochs=2, batch_size=64)
        except Exception:
            self.fail("ReachAbsolute drill raises error.")

    def test_drill_freereach(self):
        """Test drilling of free reach agent."""

        try:
            wrappers = [StateNormalizationTransformer, RewardNormalizationTransformer]
            env = make_env("FreeReachAbsolute-v0", reward_config=None, transformers=wrappers)
            build_models = get_model_builder(model="shadow", model_type="lstm", shared=False)
            agent = PPOAgent(build_models, env, workers=2, horizon=128, distribution=BetaPolicyDistribution(env))
            agent.drill(n=2, epochs=2, batch_size=64)
        except Exception:
            self.fail("FreeReachAbsolute drill raises error.")

    def test_classic_control(self):
        for env_name in ["CartPole-v1", "Acrobot-v1", "Pendulum-v1", "MountainCar-v0"]:
            try:
                _test_drill(env_name)
            except Exception:
                self.fail("Continuous drill raises error.")

    def test_robotic_control(self):
        for env_name in ["Ant-v4", "Humanoid-v4"]:
            try:
                _test_drill(env_name)
            except Exception:
                self.fail("Continuous drill raises error.")


if __name__ == '__main__':
    unittest.main()
