import argparse
import colorsys
import json
import os
import re
from typing import Dict, Tuple, List

import gym
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np

from agent.policies import get_distribution_by_short_name
from agent.ppo import PPOAgent
from models import build_ffn_models, build_rnn_models
import configs
from configs import derive_config


def test_environment(env_name, settings, model_type: str, n: int, init_ray: bool = True):
    """Train on an environment and return the reward history."""
    env = gym.make(env_name)

    if model_type == "ffn":
        build_models = build_ffn_models
    elif model_type == "rnn":
        build_models = build_rnn_models
    else:
        raise ValueError("Unknown Model Type.")

    if settings["distribution"] is not None:
        distribution = get_distribution_by_short_name(settings["distribution"])(env)
    else:
        distribution = None

    # set up the agent and a reporting module
    agent = PPOAgent(build_models, env, horizon=settings["horizon"], workers=settings["workers"],
                     learning_rate=settings["lr_pi"], discount=settings["discount"], lam=settings["lam"],
                     clip=settings["clip"], c_entropy=settings["c_entropy"], c_value=settings["c_value"],
                     gradient_clipping=settings["grad_norm"], clip_values=settings["clip_values"],
                     tbptt_length=settings["tbptt"], distribution=distribution)

    # train
    agent.drill(n=n, epochs=settings["epochs"], batch_size=settings["batch_size"], save_every=0, separate_eval=False,
                ray_is_initialized=not init_ray, stop_early=settings["stop_early"])
    env.close()

    return agent.cycle_reward_history


def lighten_color(color, amount=0.5):
    """Lightens the given color by multiplying (1-luminosity) by the given amount."""
    try:
        c = mc.cnames[color]
    except Exception:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))

    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


if __name__ == '__main__':
    all_envs = [e.id for e in list(gym.envs.registry.all())]
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    parser = argparse.ArgumentParser(description="Perform a comparative benchmarking experiment. Given one task, a "
                                                 "number of agents is trained with different configurations, "
                                                 "for instance comparing different architectures.")
    parser.add_argument("env", type=str, choices=all_envs, help="environment in which the configurations are compared")
    parser.add_argument("name", type=str, nargs="?", default=None, help="the name of the experiment, used for data "
                                                                        "saving, will default to a combination of env "
                                                                        "and configs")
    parser.add_argument("--repetitions", "-r", type=int, help="number of repetitions of each configuration for means"
                        , default=10)
    parser.add_argument("--cycles", "-i", type=int, help="number of cycles during one drill", default=100)
    parser.add_argument("--configs", "-c", type=str, nargs="+", help="a list of configurations to be compared")
    parser.add_argument("--stop-early", "-e", action="store_true", help="allow independent early stopping")
    args = parser.parse_args()

    configurations = {n: derive_config(getattr(configs, n), {"stop_early": args.stop_early}) for n in args.configs}
    if args.name is None:
        args.name = re.sub("-v[0-9]", "", args.env) + "_" + "_".join(args.configs)

    results: Dict[str, Tuple[List, List]] = {}
    should_init = True
    for conf_name, config in configurations.items():
        reward_histories = []
        for i in range(args.repetitions):
            print(f"\nRepetition {i + 1}/{args.repetitions} in environment {args.env} with model {config['model']}.")
            reward_histories.append(test_environment(args.env, config, model_type=config["model"],
                                                     n=args.cycles, init_ray=should_init))
            should_init = False

            means = np.mean(reward_histories, axis=0)
            stdevs = np.std(reward_histories, axis=0)
            results.update({conf_name: (means.tolist(), stdevs.tolist())})

            with open(f"docs/benchmarks/{args.name}.json", "w") as f:
                json.dump(results, f, indent=2)

        x = list(range(1, args.cycles + 1))
        plt.plot(x, results[conf_name][0], 'k-')
        plt.fill_between(x,
                         np.array(results[conf_name][0]) - np.array(results[conf_name][1]),
                         np.array(results[conf_name][0]) + np.array(results[conf_name][1]),
                         label=f"{args.env} ({conf_name})")

    plt.legend()
    plt.savefig(f"docs/benchmarks/benchmarking_{args.name}.pdf", format="pdf")
