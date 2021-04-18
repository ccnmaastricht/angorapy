import argparse
import json
import os

import gym
import numpy as np

import configs
from configs import derive_config
from train import run_experiment
from common.const import PATH_TO_BENCHMARKS
from utilities.statistics import increment_mean_var

if __name__ == '__main__':
    all_envs = [e.worker_id for e in list(gym.envs.registry.all())]
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
    parser.add_argument("--cycles", "-i", type=int, help="number of cycles during one drill", default=None)
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
        config["iterations"] = args.cycles if args.cycles is not None else config["iterations"]
        config["config"] = conf_name
        config["eval"] = True

        conf_up_counter = 1
        original_conf_name = conf_name
        while conf_name in benchmark_dict["results"]:
            conf_compatible = args.cycles == len(benchmark_dict["results"][conf_name]["means"])

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
            reward_history = np.array(run_experiment(
                args.env, config, verbose=False).cycle_reward_history)
            should_init = False

            current_n = benchmark_dict["results"][conf_name]["n"]
            if current_n == 0:
                means, var = reward_history, np.zeros_like(reward_history)
                mean_max, var_max = np.max(reward_history), np.array(0)
            else:
                means, var = increment_mean_var(np.array(benchmark_dict["results"][conf_name]["means"]),
                                                np.array(benchmark_dict["results"][conf_name]["var"]),
                                                reward_history,
                                                np.zeros_like(reward_history),
                                                current_n)
                mean_max, var_max = increment_mean_var(benchmark_dict["results"][conf_name]["mean_max"],
                                                       benchmark_dict["results"][conf_name]["var_max"],
                                                       np.max(reward_history),
                                                       np.array(0),
                                                       current_n)

            benchmark_dict["results"][conf_name].update({
                "n": benchmark_dict["results"][conf_name]["n"] + 1,

                # mean/var per cycle
                "means": means.tolist(),
                "var": var.tolist(),

                # max cycle performance
                "mean_max": mean_max.item(),
                "var_max": var_max.item()
            })

            with open(results_path, "w") as f:
                json.dump(benchmark_dict, f, indent=2)
