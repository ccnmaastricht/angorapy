import colorsys
import json
import os
from typing import Dict, Tuple, List

import gym
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np

from agent.ppo import PPOAgent
from models import build_ffn_models, build_rnn_models
from utilities import configs


def test_environment(env_name, settings, model_type: str, n: int, init_ray: bool = True):
    """Train on an environment and return the reward history."""
    env = gym.make(env_name)

    if model_type == "ffn":
        build_models = build_ffn_models
    elif model_type == "rnn":
        build_models = build_rnn_models
    else:
        raise ValueError("Unknown Model Type.")

    # set up the agent and a reporting module
    agent = PPOAgent(build_models, env, horizon=settings["horizon"], workers=settings["workers"],
                     learning_rate=settings["lr_pi"], discount=settings["discount"],
                     clip=settings["clip"], c_entropy=settings["c_entropy"], c_value=settings["c_value"],
                     lam=settings["lam"],
                     gradient_clipping=settings["grad_norm"], clip_values=settings["no_value_clip"],
                     tbptt_length=settings["tbptt"])

    # train
    agent.drill(n=n, epochs=settings["epochs"], batch_size=settings["batch_size"], save_every=0, separate_eval=False,
                ray_already_initialized=not init_ray)
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
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    experiment_name = "half_cheetah"
    config = configs.mujoco
    benchmarking_settings = [("HalfCheetah-v2", "ffn")]  # , ("HalfCheetah-v2", "rnn")]
    n_iterations = 100
    repetitions = 10

    results: Dict[str, Tuple[List, List]] = {}
    should_init = True
    for env_name, model_type in benchmarking_settings:
        identifier = str((env_name, model_type))

        reward_histories = []
        for i in range(repetitions):
            print(f"Repetition {i + 1}/{repetitions} in environment {env_name} with model {model_type}.")
            reward_histories.append(test_environment(env_name, config, model_type=model_type,
                                                     n=n_iterations, init_ray=should_init))
            should_init = False

            means = np.mean(reward_histories, axis=0)
            stdevs = np.std(reward_histories, axis=0)
            results.update({identifier: (means.tolist(), stdevs.tolist())})

            with open(f"docs/benchmarks/{experiment_name}.json", "w") as f:
                json.dump(results, f, indent=2)

        x = list(range(1, n_iterations + 1))
        plt.plot(x, results[identifier][0], 'k-')
        plt.fill_between(x,
                         np.array(results[identifier][0]) - np.array(results[identifier][1]),
                         np.array(results[identifier][0]) + np.array(results[identifier][1]),
                         label=f"{env_name} ({model_type})")

    plt.legend()
    plt.savefig(f"docs/figures/benchmarking_{experiment_name}.pdf", format="pdf")
