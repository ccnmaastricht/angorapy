import colorsys

import gym
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np

from agent.ppo import PPOAgent
from models import build_ffn_distinct_models, build_rnn_distinct_models
from utilities import configs


def test_environment(env_name, settings, model_type: str, n: int, init_ray: bool = True):
    """Train on an environment and return the reward history."""
    env = gym.make(env_name)

    if model_type == "ffn":
        build_models = build_ffn_distinct_models
    elif model_type == "rnn":
        build_models = build_rnn_distinct_models
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
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    benchmarking_settings = [("CartPole-v1", "ffn"), ("CartPole-v1", "rnn")]
    config = configs.discrete
    n_iterations = 40
    repetitions = 10

    colors = ["blue", "red"]

    results = {}
    should_init = True
    for i, (env_name, model_type) in enumerate(benchmarking_settings):
        reward_histories = []
        for j in range(repetitions):
            print(f"Repetition {j + 1}/{repetitions} in environment {env_name}.")
            reward_histories.append(test_environment(env_name, config, model_type=model_type,
                                                     n=n_iterations, init_ray=should_init))
            should_init = False

        x = list(range(1, n_iterations + 1))
        means = np.mean(reward_histories, axis=0)
        stdevs = np.std(reward_histories, axis=0)
        results.update({env_name: (means, stdevs)})

        plt.fill_between(x, means - stdevs, means + stdevs, label=f"{env_name} ({model_type})",
                         color=lighten_color(colors[i], 0.4))
        plt.plot(x, means, 'k-', color=lighten_color(colors[i], 1.2))

    plt.legend()
    plt.savefig("../docs/figures/benchmarking_rnn_fnn_cartpole.pdf", format="pdf")
    plt.show()
