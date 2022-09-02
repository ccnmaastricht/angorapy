"""Example script showcasing how to train an agent."""
import os

from angorapy.analysis.investigation import Investigator

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    from mpi4py import MPI
    is_root = MPI.COMM_WORLD.rank == 0
except:
    is_root = True

from angorapy.agent.ppo_agent import PPOAgent
from angorapy.common.policies import BetaPolicyDistribution
from angorapy.common.transformers import RewardNormalizationTransformer, StateNormalizationTransformer
from angorapy.common.wrappers import make_env

from angorapy.models import get_model_builder

# For most environments, PPO needs to normalize states and rewards; to add this functionality we wrap the environment
# with transformers fulfilling this task. You can also add your own custom transformers this way.
wrappers = [StateNormalizationTransformer, RewardNormalizationTransformer]
env = make_env("HumanoidVisualManipulateBlock-v0", reward_config=None, transformers=wrappers)

distribution = BetaPolicyDistribution(env)
build_models = get_model_builder(model="shadow", model_type="gru", blind=False, shared=False)

agent = PPOAgent(build_models, env, horizon=1024, workers=12, distribution=distribution)

agent.drill(n=10, epochs=3, batch_size=64)
Investigator.from_agent(agent).render_episode(agent.env, act_confidently=True)
