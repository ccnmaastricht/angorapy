import os

import pytest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from angorapy.agent.ppo_agent import PPOAgent
from angorapy.common.policies import BetaPolicyDistribution, MultiCategoricalPolicyDistribution
from angorapy.common.postprocessors import RewardNormalizer, StateNormalizer
from angorapy import make_task

from angorapy.models import get_model_builder

try:
    from mpi4py import MPI

    is_root = MPI.COMM_WORLD.rank == 0
except:
    is_root = True

def _test_drill(env_name, model_builder=None, ):
    """Test drilling an agent."""
    wrappers = [StateNormalizer, RewardNormalizer]
    env = make_task(env_name, reward_config=None, postprocessors=wrappers)
    if model_builder is None:
        build_models = get_model_builder(model="simple", model_type="ffn", shared=False)
    else:
        build_models = model_builder
    agent = PPOAgent(build_models, env, workers=2, horizon=max(128, env.spec.max_episode_steps))
    agent.drill(n=2, epochs=2, batch_size=64)


def test_drill_continuous():
    """Test drilling of continuous agent (LunarLanderContinuous)."""

    try:
        _test_drill("LunarLanderContinuous-v2")
    except Exception:
        pytest.fail("Continuous drill raises error.")


def test_drill_discrete():
    """Test drilling of discrete agent (LunarLander)."""

    try:
        _test_drill("LunarLander-v2")
    except Exception:
        pytest.fail("Discrete drill raises error.")


def test_drill_manipulate_multicategorical():
    """Test drilling of discrete agent (LunarLander)."""

    try:
        wrappers = [StateNormalizer, RewardNormalizer]
        env = make_task("ManipulateBlockDiscreteAsynchronous-v0", reward_config=None, postprocessors=wrappers)
        build_models = get_model_builder(model="shadow", model_type="lstm", shared=False, blind=True)
        agent = PPOAgent(
            build_models,
            env,
            workers=2,
            horizon=128,
            distribution=MultiCategoricalPolicyDistribution(env)
        )

        agent.drill(n=2, epochs=2, batch_size=64)
    except Exception:
        pytest.fail("ManipulateBlockDiscreteAsynchronous drill raises error.")


def test_drill_manipulate_continuous():
    """Test drilling of discrete agent (LunarLander)."""

    try:
        wrappers = [StateNormalizer, RewardNormalizer]
        env = make_task("ManipulateBlockAsynchronous-v0", reward_config=None, postprocessors=wrappers)
        build_models = get_model_builder(model="shadow", model_type="lstm", shared=False)
        agent = PPOAgent(build_models, env, workers=2, horizon=128, distribution=BetaPolicyDistribution(env))
        agent.drill(n=2, epochs=2, batch_size=64)
    except Exception:
        pytest.fail("ManipulateBlockDiscreteAsynchronous drill raises error.")


def test_drill_reach():
    """Test drilling of discrete agent (LunarLander)."""

    try:
        wrappers = [StateNormalizer, RewardNormalizer]
        env = make_task("ReachAbsolute-v0", reward_config=None, postprocessors=wrappers)
        build_models = get_model_builder(model="shadow", model_type="lstm", shared=False)
        agent = PPOAgent(build_models, env, workers=2, horizon=128, distribution=BetaPolicyDistribution(env))
        agent.drill(n=2, epochs=2, batch_size=64)
    except Exception:
        pytest.fail("ReachAbsolute drill raises error.")


def test_drill_freereach():
    """Test drilling of free reach agent."""

    try:
        wrappers = [StateNormalizer, RewardNormalizer]
        env = make_task("FreeReachAbsolute-v0", reward_config=None, postprocessors=wrappers)
        build_models = get_model_builder(model="shadow", model_type="lstm", shared=False)
        agent = PPOAgent(build_models, env, workers=2, horizon=128, distribution=BetaPolicyDistribution(env))
        agent.drill(n=2, epochs=2, batch_size=64)
    except Exception:
        pytest.fail("FreeReachAbsolute drill raises error.")


def test_classic_control():
    for env_name in ["CartPole-v1", "Acrobot-v1", "Pendulum-v1", "MountainCar-v0"]:
        try:
            _test_drill(env_name)
        except Exception:
            pytest.fail("Continuous drill raises error.")


def test_robotic_control():
    for env_name in ["Ant-v4", "Humanoid-v4"]:
        try:
            _test_drill(env_name)
        except Exception:
            pytest.fail("Continuous drill raises error.")
