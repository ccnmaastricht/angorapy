import argparse
import json
import os

import gym
import numpy as np

import configs
from agent.policies import get_distribution_by_short_name
from agent.ppo import PPOAgent
from configs import derive_config
from models import build_ffn_models, build_rnn_models, get_model_builder
from utilities.const import PATH_TO_BENCHMARKS
from utilities.statistics import increment_mean_var
from utilities.util import env_extract_dims
from utilities.wrappers import CombiWrapper, StateNormalizationWrapper, RewardNormalizationWrapper, SkipWrapper


def test_environment(env_name, settings, model_type: str, n: int, init_ray: bool = True):
    """Train on an environment and return the reward history."""
    env = gym.make(env_name)
    state_dim, action_dim = env_extract_dims(env)

    # model
    if model_type == "ffn":
        build_models = get_model_builder(model_type=settings.model, shared=settings.shared)
    else:
        raise ValueError("Unknown Model Type.")

    # distribution
    if settings["distribution"] is not None:
        distribution = get_distribution_by_short_name(settings["distribution"])(env)
    else:
        distribution = None

    # preprocessor
    preprocessor = CombiWrapper(
        [StateNormalizationWrapper(state_dim) if not settings["no_state_norming"] else SkipWrapper(),
         RewardNormalizationWrapper() if not settings["no_reward_norming"] else SkipWrapper()])

    # set up the agent and a reporting module
    agent = PPOAgent(build_models, env, horizon=settings["horizon"], workers=settings["workers"],
                     learning_rate=settings["lr_pi"], discount=settings["discount"], lam=settings["lam"],
                     clip=settings["clip"], c_entropy=settings["c_entropy"], c_value=settings["c_value"],
                     gradient_clipping=settings["grad_norm"], clip_values=settings["clip_values"],
                     tbptt_length=settings["tbptt"], distribution=distribution, preprocessor=preprocessor)

    # train
    agent.drill(n=n, epochs=settings["epochs"], batch_size=settings["batch_size"], save_every=0, separate_eval=False,
                ray_is_initialized=not init_ray, stop_early=settings["stop_early"], save_best=True)
    env.close()

    return agent.cycle_reward_history


if __name__ == '__main__':
    all_envs = [e.id for e in list(gym.envs.registry.all())]
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    parser = argparse.ArgumentParser(description="Perform a comparative benchmarking experiment. Given one task, a "
                                                 "number of agents is trained with different configurations, "
                                                 "for instance comparing different architectures.")
    parser.add_argument("env", type=str, choices=all_envs, help="environment in which the configurations are compared")
    parser.add_argument("name", type=str, nargs="?", default="default",
                        help="the name of the experiment, used for data "
                             "saving, will default to a combination of env "
                             "and configs")
    parser.add_argument("--repetitions", "-r", type=int, help="number of repetitions of each configuration for means"
                        , default=10)
    parser.add_argument("--cycles", "-i", type=int, help="number of cycles during one drill", default=100)
    parser.add_argument("--configs", "-c", type=str, nargs="+", help="a list of configurations to be compared")
    parser.add_argument("--stop-early", "-e", action="store_true", help="allow independent early stopping")
    args = parser.parse_args()

    configurations = {n: derive_config(getattr(configs, n), {"stop_early": args.stop_early}) for n in args.configs}
    results_file_name = f"{args.name}_{args.env}.json"
    results_path = f"{PATH_TO_BENCHMARKS}/{results_file_name}"

    benchmark_dict = {
        "results": {}
    }

    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            benchmark_dict = json.load(f)

    benchmark_dict.update({
        "meta": {
            "reward_threshold": gym.make(args.env).spec.reward_threshold,
        },
    })

    should_init = True
    for conf_name, config in configurations.items():
        conf_up_counter = 1
        original_conf_name = conf_name
        while conf_name in benchmark_dict["results"]:
            conf_compatible = (
                    args.cycles == len(benchmark_dict["results"][conf_name]["means"])
                    # and config == benchmark_dict["results"][conf_name]["config"]
            )

            if conf_compatible:
                print(f"Found compatible config; Extending {conf_name}.")
                break

            conf_name = f"{original_conf_name}_{conf_up_counter}"
        else:
            print(f"No compatibles to {original_conf_name} benchmarked, adding new condition {conf_name}.")
            benchmark_dict["results"].update({
                conf_name: {
                    "n": 0
                }
            })

        benchmark_dict["results"][conf_name].update({
            "config": config
        })

        for i in range(args.repetitions):
            print(f"\nRepetition {i + 1}/{args.repetitions} in environment {args.env} with config {conf_name}.")
            reward_history = np.array(test_environment(args.env, config, model_type=config["model"],
                                                       n=args.cycles, init_ray=should_init))
            should_init = False

            current_n = benchmark_dict["results"][conf_name]["n"]
            if current_n == 0:
                means, var = reward_history, np.zeros_like(reward_history)
            else:
                means, var = increment_mean_var(np.array(benchmark_dict["results"][conf_name]["means"]),
                                                np.array(benchmark_dict["results"][conf_name]["var"]),
                                                reward_history,
                                                np.zeros_like(reward_history),
                                                current_n)

            benchmark_dict["results"][conf_name].update({
                "n": benchmark_dict["results"][conf_name]["n"] + 1,
                "means": means.tolist(),
                "var": var.tolist()
            })

            with open(results_path, "w") as f:
                json.dump(benchmark_dict, f, indent=2)
