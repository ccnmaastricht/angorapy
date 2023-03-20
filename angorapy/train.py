import logging
import re

logging.getLogger("requests").setLevel(logging.WARNING)

import sys
import os

os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pprint
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from angorapy.utilities.defaults import autoselect_distribution

import distance
import numpy as np

import tensorflow as tf
from tensorflow.keras import mixed_precision

import argparse
import logging

import argcomplete
from gym.spaces import Box, Discrete, MultiDiscrete

from angorapy.configs import hp_config
from angorapy.common.policies import get_distribution_by_short_name
from angorapy.models import get_model_builder, MODELS_AVAILABLE
from angorapy.common.const import COLORS
from angorapy.utilities.monitoring import Monitor
from angorapy.utilities.util import env_extract_dims
from angorapy.common.wrappers import make_env
from angorapy.common.transformers import StateNormalizationTransformer, RewardNormalizationTransformer
from angorapy.agent.ppo_agent import PPOAgent

from angorapy.environments import *

from mpi4py import MPI


class InconsistentArgumentError(Exception):
    """Arguments given to a process were inconsistent."""
    pass


def run_experiment(environment, settings: dict, verbose=True, use_monitor=False):
    """Run an experiment with the given settings ."""
    if settings["cpu"]:
        print("Deactivating GPU")
        tf.config.set_visible_devices([], "GPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    mpi_rank = MPI.COMM_WORLD.rank
    is_root = mpi_rank == 0

    # sanity checks and warnings for given parameters
    if settings["preload"] is not None and settings["load_from"] is not None:
        raise InconsistentArgumentError("You gave both a loading from a pretrained component and from another "
                                        "agent state. This cannot be resolved.")

    # determine relevant transformers
    wrappers = []
    wrappers.append(StateNormalizationTransformer) if not settings["no_state_norming"] else None
    wrappers.append(RewardNormalizationTransformer) if not settings["no_reward_norming"] else None

    # setup environment and extract and report information
    env = make_env(environment,
                   reward_config=settings["rcon"],
                   transformers=wrappers,
                   render_mode="rgb_array" if re.match(".*[Vv]is(ion|ual).*", environment) else None)
    state_dim, number_of_actions = env_extract_dims(env)

    if env.spec.max_episode_steps is not None and env.spec.max_episode_steps > settings["horizon"] and not settings[
        "eval"] and is_root:
        logging.warning("Careful! Your horizon is lower than the max environment steps, "
                        "this will most likely skew stats heavily.")

    # choose and make policy distribution
    if settings["distribution"] is None:
        distribution = autoselect_distribution(env)
    else:
        distribution = get_distribution_by_short_name(settings["distribution"])(env)

    # setting appropriate model building function
    blind = not("vision" in list(env.observation_space["observation"].keys())
                and len(env.observation_space["observation"]["vision"].shape) > 1)
    build_models = get_model_builder(model=settings["architecture"], model_type=settings["model"], shared=settings["shared"],
                                     blind=blind)

    # announce experiment
    bc, ec, wn = COLORS["HEADER"], COLORS["ENDC"], COLORS["WARNING"]
    if verbose and is_root:
        env_action_space_type = "continuous" if isinstance(env.action_space, Box) else "discrete"
        env_observation_space_type = "continuous" if isinstance(env.observation_space, Box) else "discrete"
        env_name = env.unwrapped.spec.id

        print(f"-----------------------------------------\n"
              f"{wn}Learning the Task{ec}: {bc}{env_name}{ec}\n"
              f"{bc}{state_dim}{ec}-dimensional states ({bc}{env_observation_space_type}{ec}) "
              f"and {bc}{number_of_actions}{ec} actions ({bc}{env_action_space_type}{ec}).\n"
              f"Config: {settings['pcon']}\n"
              f"Model: {build_models.__name__}\n"
              f"Distribution: {settings['distribution']}\n"
              f"-----------------------------------------\n")

        print(f"{wn}HyperParameters{ec}: {pprint.pformat(settings)}\n")

    if settings["load_from"] is not None:
        if verbose and is_root:
            print(f"{wn}Loading{ec} from state {settings['load_from']}")
        agent = PPOAgent.from_agent_state(settings["load_from"], from_iteration="last",
                                          n_optimizers=settings["n_optimizers"])
    else:
        # set up the agent and a reporting module
        agent = PPOAgent(build_models, env, horizon=settings["horizon"], workers=settings["workers"],
                         learning_rate=settings["lr_pi"], discount=settings["discount"], lam=settings["lam"],
                         clip=settings["clip"], c_entropy=settings["c_entropy"], c_value=settings["c_value"],
                         gradient_clipping=settings["grad_norm"], clip_values=settings["clip_values"],
                         tbptt_length=settings["tbptt"], lr_schedule=settings["lr_schedule"], distribution=distribution,
                         reward_configuration=settings["rcon"], debug=settings["debug"],
                         pretrained_components=None if settings["preload"] is None else [settings["preload"]],
                         n_optimizers=settings["n_optimizers"])

        if is_root:
            print(f"{wn}Created agent{ec} with ID {bc}{agent.agent_id}{ec}")

        # load pretrained components
        if settings["component_dir"] is not None:
            agent.load_components(settings["component_dir"])

        # freeze components
        if settings["freeze_components"] is not None:
            for component in settings["freeze_components"]:
                agent.freeze_component(component)

    if len(tf.config.list_physical_devices('GPU')) > 0:
        agent.set_gpu(not settings["cpu"])
    else:
        agent.set_gpu(False)

    monitor = None
    if use_monitor and is_root:
        monitor = Monitor(agent,
                          agent.env,
                          frequency=settings["monitor_frequency"],
                          gif_every=settings["gif_every"],
                          id=agent.agent_id,
                          iterations=settings["iterations"],
                          config_name=settings["pcon"],
                          experiment_group=settings["experiment_group"])

    try:
        agent.drill(n=settings["iterations"], epochs=settings["epochs"], batch_size=settings["batch_size"],
                    monitor=monitor, save_every=settings["save_every"], separate_eval=settings["eval"],
                    stop_early=settings["stop_early"], radical_evaluation=settings["radical_evaluation"])
    except KeyboardInterrupt:
        print("test")
    except Exception:
        if mpi_rank == 0:
            traceback.print_exc()
    finally:
        if mpi_rank == 0:
            agent.finalize()

    agent.save_agent_state()
    env.close()

    return agent


if __name__ == "__main__":
    tf.get_logger().setLevel('INFO')
    all_envs = [e.id for e in list(gym.envs.registry.values())]

    # parse commandline arguments
    parser = argparse.ArgumentParser(description="Train a PPO Agent on some task.")

    # general parameters
    parser.add_argument("env", nargs='?', type=str, default="ReachAbsolute-v0", help="the target gym environment")
    parser.add_argument("--architecture", choices=MODELS_AVAILABLE, default="simple",
                        help="architecture of the policy")
    parser.add_argument("--model", choices=["ffn", "rnn", "lstm", "gru"], default="ffn",
                        help=f"model type if architecture allows for choices")
    parser.add_argument("--distribution", type=str, default=None,
                        choices=["categorical", "gaussian", "beta", "rbeta", "multi-categorical"])
    parser.add_argument("--shared", action="store_true",
                        help=f"make the model share part of the network for policy and value")
    parser.add_argument("--iterations", type=int, default=5000, help=f"number of iterations before training ends")

    # meta arguments
    parser.add_argument("--pcon", type=str, default=None, help="config name (utilities/hp_config.py) to be loaded")
    parser.add_argument("--rcon", type=str, default=None,
                        help="config (utilities/reward_config.py) of the reward function")
    parser.add_argument("--experiment-group", type=str, default="default", help="experiment group identifier")
    parser.add_argument("--cpu", action="store_true", help=f"use cpu only")
    parser.add_argument("--sequential", action="store_true", help=f"run worker sequentially workers")
    parser.add_argument("--load-from", type=int, default=None, help=f"load from given agent id")
    parser.add_argument("--preload", type=str, default=None, help=f"load visual component weights from pretraining")
    parser.add_argument("--component-dir", type=str, default=None, help=f"path to pretrained components")
    parser.add_argument("--export-file", type=int, default=None, help=f"save policy to be loaded in workers into file")
    parser.add_argument("--eval", action="store_true", help=f"evaluate additionally to have at least 5 eps")
    parser.add_argument("--radical-evaluation", action="store_true", help=f"only record stats from seperate evaluation")
    parser.add_argument("--save-every", type=int, default=0, help=f"save agent every given number of iterations")
    parser.add_argument("--monitor-frequency", type=int, default=1, help=f"update the monitor every n iterations.")
    parser.add_argument("--gif-every", type=int, default=0, help=f"make a gif every n iterations.")
    parser.add_argument("--debug", action="store_true", help=f"run in debug mode (eager mode)")
    parser.add_argument("--no-monitor", action="store_true", help="dont use a monitor")
    parser.add_argument("--freeze-components", type=str, nargs="+", help=f"Components to freeze the weights of.")
    parser.add_argument("--n-optimizers", type=int, default=None, help=f"number of optimizers; "
                                                                       f"default is all GPUs or CPUs (if no GPUs found)")

    # gathering parameters
    parser.add_argument("--workers", type=int, default=8, help=f"the number of workers exploring the environment")
    parser.add_argument("--horizon", type=int, default=2048, help=f"number of time steps one worker generates per cycle")
    parser.add_argument("--discount", type=float, default=0.99, help=f"discount factor for future rewards")
    parser.add_argument("--lam", type=float, default=0.97, help=f"lambda parameter in the GAE algorithm")
    parser.add_argument("--no-state-norming", action="store_true", help=f"do not normalize states")
    parser.add_argument("--no-reward-norming", action="store_true", help=f"do not normalize rewards")

    # optimization parameters
    parser.add_argument("--epochs", type=int, default=3, help=f"the number of optimization epochs in each cycle")
    parser.add_argument("--batch-size", type=int, default=64, help=f"minibatch size during optimization")
    parser.add_argument("--lr-pi", type=float, default=1e-3, help=f"learning rate of the policy")
    parser.add_argument("--lr-schedule", type=str, default=None, choices=[None, "exponential"],
                        help=f"lr schedule type")
    parser.add_argument("--clip", type=float, default=0.2, help=f"clipping range around 1 for the objective function")
    parser.add_argument("--c-entropy", type=float, default=0.01, help=f"entropy factor in objective function")
    parser.add_argument("--c-value", type=float, default=1, help=f"value factor in objective function")
    parser.add_argument("--tbptt", type=int, default=16, help=f"length of subsequences in truncated BPTT")
    parser.add_argument("--grad-norm", type=float, default=0.5, help=f"norm for gradient clipping, 0 deactivates")
    parser.add_argument("--clip-values", action="store_true", help=f"clip value objective")
    parser.add_argument("--stop-early", action="store_true", help=f"stop early if threshold of env was surpassed")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    is_root = rank == 0

    # read arguments
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.env not in all_envs:
        if is_root:
            indices = np.argsort([distance.levenshtein(w, args.env) for w in all_envs])[:3]
            print(f"Unknown environment {args.env}. Did you mean one of {[all_envs[i] for i in indices]}")
        exit()

    # if config is given load it as default, then overwrite with any goal given parameters
    if args.pcon is not None:
        try:
            parser.set_defaults(**getattr(hp_config, args.pcon))
            args = parser.parse_args()
            if is_root:
                print(f"Loaded Config {args.pcon}.")
        except AttributeError as err:
            raise ImportError("Cannot find config under given name. Does it exist in utilities/hp_config.py?")

    if args.debug:
        tf.config.run_functions_eagerly(True)
        if is_root:
            logging.warning("YOU ARE RUNNING IN DEBUG MODE!")

    try:
        run_experiment(args.env, vars(args), use_monitor=not args.no_monitor)
    except Exception as e:
        if rank == 0:
            traceback.print_exc()
