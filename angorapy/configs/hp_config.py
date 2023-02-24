"""Predefined configurations for different (groups of) envs."""


def make_config(batch_size=128, c_entropy=0.01, c_value=1, clip=0.2, cpu=False, debug=False, discount=0.99,
                env='CartPole-v1', epochs=3, eval=False, export_file=None, grad_norm=0.5, horizon=2024,
                iterations=100, lam=0.97, load_from=None, lr_pi=0.001, clip_values=False, save_every=0,
                workers=8, tbptt: int = 16, lr_schedule=None, no_state_norming=False, no_reward_norming=False,
                model="ffn", early_stopping=False, distribution=None, shared=False, preload=None, sequential=False,
                architecture="simple", radical_evaluation=False, redis_ip=None):
    """Make a config from scratch."""
    return dict(**locals())


def derive_config(original: dict, overrules: dict):
    """Make a new config with another config as base, overruling a given set of parameters."""
    derived = original.copy()
    for k, v in overrules.items():
        derived[k] = v
    return derived


test = make_config(
    batch_size=64,
    horizon=64,
    c_entropy=0.01,
    lr_pi=0.0003,
    epochs=2,
    clip=0.2,
    lam=0.95,
    discount=0.99,
    grad_norm=0.5,
    iterations=100,
    workers=2,
    clip_values=False
)

# DISCRETE: general hp settings that solve
#   - CartPole (turn of state and reward normalization though)
#   - Acrobot
#   - MountainCar (norming is absolutely crucial here

discrete = make_config(
    batch_size=128,
    horizon=2048,
    c_entropy=0.01,
    lr_pi=0.0003,
    epochs=10,
    clip=0.2,
    lam=0.95,
    discount=0.99,
    grad_norm=0.5,
    iterations=100,
    workers=8,
    clip_values=False
)

discrete_no_ent = derive_config(discrete, {"c_entropy": 0.0})
discrete_shared = derive_config(discrete, {"shared": True})
discrete_rnn = derive_config(discrete, {"model": "rnn"})
discrete_gru = derive_config(discrete, {"model": "gru"})
discrete_lstm = derive_config(discrete, {"model": "lstm"})
discrete_no_norms = derive_config(discrete, {"no_state_norming": True, "no_reward_norming": True})
discrete_no_norms_short = derive_config(discrete_no_norms, {"horizon": 512})

# CONTINUOUS DEFAULT
#   - LunarLanderContinuous
#   - BipedalWalker (use beta distribution to crack the reward threshold)

continuous = make_config(
    batch_size=512,
    horizon=2048,
    c_entropy=0.0,
    lr_pi=0.0003,
    epochs=10,
    clip=0.2,
    lam=0.95,
    discount=0.99,
    grad_norm=0.5,
    iterations=100,
    workers=8,
    clip_values=False
)

continuous_shared = derive_config(continuous, {"shared": True})
continuous_rnn = derive_config(continuous, {"model": "rnn"})
continuous_gru = derive_config(continuous, {"model": "gru"})
continuous_lstm = derive_config(continuous, {"model": "lstm"})
continuous_beta = derive_config(continuous, {"distribution": "beta"})
continuous_no_norms = derive_config(continuous, {"no_state_norming": True,
                                                 "no_reward_norming": True})

continuous_with_ent = derive_config(continuous, {"c_entropy": 0.01})
continuous_beta_with_ent = derive_config(continuous_beta, {"c_entropy": 0.01})

continuous_beta_lstm = derive_config(continuous_beta, {"model": "lstm"})
continuous_beta_gru = derive_config(continuous_beta, {"model": "gru"})
continuous_beta_rnn = derive_config(continuous_beta, {"model": "rnn"})

# PENDULUM (for, well, Pendulum-v0)

pendulum = derive_config(continuous, dict(
    horizon=512,
    epochs=10,
    discount=0.99,
    distribution="gaussian"
))

pendulum_beta = derive_config(pendulum, {"distribution": "beta"})

# FROM PAPERS

beta_paper = make_config(
    # continuous with some parameters from the beta paper
    batch_size=64,
    horizon=2048,
    c_entropy=0.001,
    lr_pi=0.0003,
    epochs=10,
    clip=0.2,
    lam=0.95,
    discount=0.995,
    grad_norm=0.5,
    iterations=100,
    workers=8,
    clip_values=False
)

# MUJOCO

mujoco = make_config(
    iterations=1000000 // 2048,  # one million timesteps
    workers=1,
    batch_size=64,
    horizon=2048,
    c_entropy=0.0,
    lr_pi=0.0003,
    epochs=10,
    clip=0.2,
    lam=0.95,
    discount=0.99,
    grad_norm=0.5,
    clip_values=False
)

mujoco_beta = derive_config(mujoco, {"distribution": "beta"})
mujoco_beta_rnn = derive_config(mujoco_beta, {"model": "rnn"})
mujoco_beta_gru = derive_config(mujoco_beta, {"model": "gru"})
mujoco_beta_lstm = derive_config(mujoco_beta, {"model": "lstm"})
mujoco_vc = derive_config(mujoco, {"clip_values": True})
mujoco_beta_shared = derive_config(mujoco_beta, {"shared": True})

# ROBOSCHOOL TASKS

roboschool = make_config(
    iterations=50000000 // (36 * 512),  # 50 million timesteps
    workers=36,
    batch_size=4096,
    horizon=512,
    c_entropy=0.0,
    lr_pi=0.0003,
    lr_schedule="exponential",  # should be a linear annealing
    epochs=15,
    clip=0.2,
    lam=0.95,
    discount=0.99,
    grad_norm=0.5,
    clip_values=False,
    radical_evaluation=True,  # we need to evaluate everytime, because horizon is short than max steps, which biases
    # statistics towards failing episodes
)

roboschool_beta = derive_config(roboschool, {"distribution": "beta"})

# HAND ENVS

hand = make_config(
    iterations=50000,
    workers=16,
    batch_size=256,
    horizon=512,
    c_entropy=0.001,
    lr_pi=5e-4,
    lr_schedule="exponential",
    epochs=10,
    clip=0.2,
    lam=0.95,
    discount=0.998,
    grad_norm=0.5,
    clip_values=False,
    distribution="beta",
    architecture="shadow",
    model="gru"
)

hand_no_ent = derive_config(hand, {"c_entropy": 0.0})
hand_manipulate = derive_config(hand, {
    "c_entropy": 0.01,
    "epochs": 3
})

andrypulate = make_config(
    iterations=50000,
    workers=384,
    batch_size=384 * 32,
    horizon=2048,
    c_entropy=0.001,
    lr_pi=3e-4,
    lr_schedule="exponential",
    epochs=3,
    clip=0.2,
    lam=0.95,
    discount=0.998,
    grad_norm=0.5,
    tbptt=8,
    clip_values=False,
    distribution="multi-categorical",
    architecture="wider",
    model="lstm",
)

akkaypulate = make_config(
    iterations=50000,
    workers=396,   # akkaya et al use 400 but that would not divide by 12; on cscs, this needs 33 gpu nodes
    batch_size=256,
    horizon=4096,
    c_entropy=0.01,
    lr_pi=3e-4,
    lr_schedule="exponential",
    epochs=3,
    clip=0.2,
    lam=0.95,
    discount=0.998,
    grad_norm=0.5,
    clip_values=False,
    distribution="beta",
    architecture="wider",
    model="lstm"
)

manipulate = make_config(
    iterations=50000,
    workers=384,   # need divisibility by 12
    batch_size=384 * 32,
    horizon=2048,
    c_entropy=0.01,
    lr_pi=3e-4,
    lr_schedule="exponential",
    epochs=3,
    clip=0.2,
    lam=0.95,
    discount=0.998,
    grad_norm=0.5,
    clip_values=False,
    distribution="beta",
    architecture="wider",
    model="lstm"
)

manipulate_discrete = derive_config(manipulate, dict(
    distribution="multi-categorical",
    c_entropy=0.001,
))

# RECOMMENDED CONFIGS FOR ENVs

recommended_config = dict(
    **dict.fromkeys(
        ["CartPole-v0", "CartPole-v1"], discrete_no_norms
    ), **dict.fromkeys(
        ["Acrobot-v1", "MountainCar-v0", "LunarLander-v2"], discrete
    ), **dict.fromkeys(
        ["Pendulum-v1"], pendulum_beta
    ), **dict.fromkeys(
        ["LunarLanderContinuous-v2", "BipedalWalker-v3", "BipedalWalkerHardcore-v2"], continuous_beta
    ), **dict.fromkeys(
        ["HalfCheetah-v2", "Hopper-v2", "InvertedPendulum-v2", "InvertedDoublePendulum-v2",
         "Swimmer-v2", "Walker2d-v2"], mujoco_beta
    ), **dict.fromkeys(
        ["Reacher-v2"], mujoco
    ), **dict.fromkeys(
        ["Humanoid-v2", "HumanoidStandup-v2"], roboschool
    ),
)

if __name__ == '__main__':
    import pandas as pd

    config_table = pd.DataFrame.from_dict(
        dict(
            discrete=discrete,
            continuous=continuous,
            pendulum=pendulum,
            mujoco=mujoco,
            roboschool=roboschool,
            hand=hand
        )
    )

    config_table = config_table.drop(
        ["env", "radical_evaluation", "preload", "sequential", "debug", "cpu", "export_file",
         "load_from", "save_every"])

    with open("../../docs/hp_table.tex", "w") as f:
        config_table.to_latex(f, bold_rows=True)
