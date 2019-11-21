import argparse
import logging
import os

import argcomplete
from gym.spaces import Box

from agent.ppo import PPOAgent
from environments import *
from models.fully_connected import build_ffn_distinct_models
from models.hybrid import build_shadow_brain
from utilities.const import COLORS
from utilities.monitoring import StoryTeller
from utilities.util import env_extract_dims

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def run_experiment(settings: argparse.Namespace):
    """Run an experiment with the given settings."""

    # setting appropriate model building function
    if settings.env == "ShadowHand-v1":
        build_models = build_shadow_brain
    else:
        build_models = build_ffn_distinct_models

    if settings.debug:
        logging.warning("YOU ARE RUNNING IN DEBUG MODE!")

    # setup environment and extract and report information
    env = gym.make(settings.env)
    state_dimensionality, number_of_actions = env_extract_dims(env)
    env_action_space_type = "continuous" if isinstance(env.action_space, Box) else "discrete"
    env_observation_space_type = "continuous" if isinstance(env.observation_space, Box) else "discrete"
    env_name = env.unwrapped.spec.id

    # announce experiment
    bc, ec, wn = COLORS["HEADER"], COLORS["ENDC"], COLORS["WARNING"]
    print(f"-----------------------------------------\n"
          f"{wn}Learning the Task{ec}: {bc}{env_name}{ec}\n"
          f"{bc}{state_dimensionality}{ec}-dimensional states ({bc}{env_observation_space_type}{ec}) "
          f"and {bc}{number_of_actions}{ec} actions ({bc}{env_action_space_type}{ec}).\n"
          f"Model: {build_models.__name__}\n"
          f"-----------------------------------------\n")

    print(f"{wn}HyperParameters{ec}: {args}\n")

    if settings.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if settings.load_from is not None:
        print(f"{wn}Loading{ec} from state {settings.load_from}")
        agent = PPOAgent.from_agent_state(settings.load_from)
    else:
        # set up the agent and a reporting module
        agent = PPOAgent(build_models, env, horizon=settings.horizon, workers=settings.workers,
                         learning_rate=settings.lr_pi, discount=settings.discount,
                         clip=settings.clip, c_entropy=settings.c_entropy, c_value=settings.c_value, lam=settings.lam,
                         gradient_clipping=settings.grad_norm, clip_values=settings.no_value_clip,
                         tbptt_length=settings.tbptt, debug=settings.debug)

        print(f"{wn}Created agent{ec} with ID {bc}{agent.agent_id}{ec}")

    agent.set_gpu(not settings.cpu)
    teller = StoryTeller(agent, env, frequency=0)

    # train
    agent.drill(n=settings.iterations, epochs=settings.epochs, batch_size=settings.batch_size, story_teller=teller,
                export=settings.export_file, save_every=settings.save_every, separate_eval=settings.eval)

    agent.save_agent_state()
    env.close()


if __name__ == "__main__":
    all_envs = [e.id for e in list(gym.envs.registry.all())]

    # parse commandline arguments
    parser = argparse.ArgumentParser(description="Train a PPO Agent on some task.")

    parser.add_argument("env", nargs='?', type=str, default="ShadowHand-v1", choices=all_envs)
    parser.add_argument("-w", "--workers", type=int, default=4, help=f"the number of workers exploring the environment")
    parser.add_argument("--epochs", type=int, default=3, help=f"the number of optimization epochs in each cycle")
    parser.add_argument("--horizon", type=int, default=1024, help=f"the number of optimization epochs in each cycle")
    parser.add_argument("-i", "--iterations", type=int, default=1000, help=f"number of iterations before training ends")
    parser.add_argument("-b", "--batch-size", type=int, default=64, help=f"minibatch size during optimization")
    parser.add_argument("--lr-pi", type=float, default=3e-4, help=f"learning rate of the policy")
    parser.add_argument("--discount", type=float, default=0.99, help=f"discount factor for future rewards")
    parser.add_argument("--lam", type=float, default=0.97, help=f"lambda parameter in the GAE algorithm")
    parser.add_argument("--clip", type=float, default=0.2, help=f"clipping range around 1 for the objective function")
    parser.add_argument("--c-entropy", type=float, default=0.01, help=f"entropy factor in objective function")
    parser.add_argument("--c-value", type=float, default=1, help=f"value factor in objective function")
    parser.add_argument("--tbptt", type=int, default=16, help=f"length of subsequences in truncated BPTT")
    parser.add_argument("--grad-norm", type=float, default=0.5, help=f"norm for gradient clipping")
    parser.add_argument("--no-value-clip", action="store_false",
                        help=f"deactivate clipping in value network's objective")

    parser.add_argument("--cpu", action="store_true", help=f"use cpu only")
    parser.add_argument("--load-from", type=int, default=None, help=f"load from given agent id")
    parser.add_argument("--export-file", type=int, default=None,
                        help=f"save/read policy to be loaded in workers into file")
    parser.add_argument("--eval", action="store_true", help=f"evaluate separately (instead of using worker experience)")
    parser.add_argument("--save_every", type=int, default=0,
                        help=f"save agent every given number of iterations (0 for no saving)")

    parser.add_argument("--debug", action="store_true", help=f"run in debug mode")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    args.debug = False

    run_experiment(args)
