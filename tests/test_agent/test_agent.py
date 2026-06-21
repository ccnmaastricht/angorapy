"""End-to-end smoke tests for the full PPO training loop (``PPOAgent.drill``).

Each test runs a couple of complete drill iterations (gather rollouts -> estimate
advantages -> optimize) on a representative task and asserts only that the loop
runs to completion without raising. This is deliberately a *smoke* suite: it does
not check learning quality, it checks that the whole pipeline — environment,
postprocessors, model construction, the gatherer, and the optimizer — fits
together for each major task family:

* continuous and discrete classic control (LunarLander, CartPole, ...);
* multi-discrete and continuous dexterous manipulation (ShadowHand);
* reach / free-reach tasks;
* MuJoCo robotic control (Ant, Humanoid).

Because they construct real models and step real simulators, these are the
slowest tests in the suite but also the highest-coverage: most integration
regressions surface here first.

Note: each test wraps the drill in ``try/except`` + ``pytest.fail`` so a failure
is reported against the specific task rather than as a raw traceback.
"""
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
    """Run two short drill iterations on ``env_name`` with state/reward normalization.

    Builds a task (wrapped in :class:`StateNormalizer` and :class:`RewardNormalizer`),
    a default feed-forward simple model unless ``model_builder`` is given, and a
    two-worker :class:`PPOAgent`, then drills for 2 cycles of 2 epochs. Used as the
    shared driver for the classic-control and robotic-control smoke tests.
    """
    wrappers = [StateNormalizer, RewardNormalizer]
    env = make_task(env_name, reward_config=None, postprocessors=wrappers)
    if model_builder is None:
        build_models = get_model_builder(model="simple", model_type="ffn", shared=False)
    else:
        build_models = model_builder
    agent = PPOAgent(build_models, env, workers=2, horizon=max(128, env.spec.max_episode_steps))
    agent.drill(n=2, epochs=2, batch_size=64)


def test_drill_continuous():
    """Full drill runs on a continuous-action task (LunarLanderContinuous, Beta/Gaussian).

    Rationale
        Exercises the continuous-control path end to end (feed-forward model,
        continuous action head, GAE, optimization) — the most common use case.
    """

    try:
        _test_drill("LunarLanderContinuous-v2")
    except Exception:
        pytest.fail("Continuous drill raises error.")


def test_drill_discrete():
    """Full drill runs on a discrete-action task (LunarLander, Categorical).

    Rationale
        Covers the discrete-action branch (categorical head, discrete action
        probability gather), which differs from the continuous path.
    """

    try:
        _test_drill("LunarLander-v2")
    except Exception:
        pytest.fail("Discrete drill raises error.")


def test_drill_manipulate_multicategorical():
    """Full drill runs on multi-discrete ShadowHand manipulation (recurrent, blind).

    Rationale
        Exercises the multi-categorical action space together with a recurrent
        ``shadow`` model and the dexterity simulator — the most complex
        discrete-control configuration.
    """

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
    """Full drill runs on continuous ShadowHand manipulation (recurrent, Beta policy).

    Rationale
        The continuous counterpart to the multi-categorical manipulation test;
        covers a recurrent ``shadow`` model with a Beta action head on the
        dexterity simulator.
    """

    try:
        wrappers = [StateNormalizer, RewardNormalizer]
        env = make_task("ManipulateBlockAsynchronous-v0", reward_config=None, postprocessors=wrappers)
        build_models = get_model_builder(model="shadow", model_type="lstm", shared=False)
        agent = PPOAgent(build_models, env, workers=2, horizon=128, distribution=BetaPolicyDistribution(env))
        agent.drill(n=2, epochs=2, batch_size=64)
    except Exception:
        pytest.fail("ManipulateBlockDiscreteAsynchronous drill raises error.")


def test_drill_reach():
    """Full drill runs on the ShadowHand Reach task (recurrent, Beta policy).

    Rationale
        Covers the goal-conditioned reach task end to end, including its
        observation structure and the recurrent shadow model.
    """

    try:
        wrappers = [StateNormalizer, RewardNormalizer]
        env = make_task("ReachAbsolute-v0", reward_config=None, postprocessors=wrappers)
        build_models = get_model_builder(model="shadow", model_type="lstm", shared=False)
        agent = PPOAgent(build_models, env, workers=2, horizon=128, distribution=BetaPolicyDistribution(env))
        agent.drill(n=2, epochs=2, batch_size=64)
    except Exception:
        pytest.fail("ReachAbsolute drill raises error.")


def test_drill_freereach():
    """Full drill runs on the ShadowHand FreeReach task (recurrent, Beta policy).

    Rationale
        Covers the free-reach variant (no fixed finger target), guarding its
        distinct goal/observation setup through a complete training iteration.
    """

    try:
        wrappers = [StateNormalizer, RewardNormalizer]
        env = make_task("FreeReachAbsolute-v0", reward_config=None, postprocessors=wrappers)
        build_models = get_model_builder(model="shadow", model_type="lstm", shared=False)
        agent = PPOAgent(build_models, env, workers=2, horizon=128, distribution=BetaPolicyDistribution(env))
        agent.drill(n=2, epochs=2, batch_size=64)
    except Exception:
        pytest.fail("FreeReachAbsolute drill raises error.")


def test_classic_control():
    """Full drill runs on the standard Gym classic-control suite.

    Rationale
        Sweeps CartPole, Acrobot, Pendulum and MountainCar in one test to cover
        the mix of discrete and continuous classic-control spaces with the
        default model, catching observation/action-space handling regressions.
    """
    for env_name in ["CartPole-v1", "Acrobot-v1", "Pendulum-v1", "MountainCar-v0"]:
        try:
            _test_drill(env_name)
        except Exception:
            pytest.fail("Continuous drill raises error.")


def test_robotic_control():
    """Full drill runs on MuJoCo locomotion tasks (Ant, Humanoid).

    Rationale
        Covers high-dimensional continuous MuJoCo control, exercising larger
        observation/action spaces than the classic-control tasks.
    """
    for env_name in ["Ant-v4", "Humanoid-v4"]:
        try:
            _test_drill(env_name)
        except Exception:
            pytest.fail("Continuous drill raises error.")
