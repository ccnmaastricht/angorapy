"""Predefined configurations for different (groups of) environments."""


def make_config(batch_size=128, c_entropy=0.01, c_value=1, clip=0.2, cpu=False, debug=False, discount=0.99,
                env='CartPole-v1', epochs=3, eval=False, export_file=None, grad_norm=0.5, horizon=1024,
                iterations=1000, lam=0.97, load_from=None, lr_pi=0.001, clip_values=False, save_every=0,
                workers=8, tbptt: int = 16, lr_schedule=None, no_state_norming=False, no_reward_norming=False,
                model="ffn", early_stopping=False, distribution=None, shared=False):
    """Make a config from scratch."""
    return dict(**locals())


def derive_config(original: dict, overrules: dict):
    """Make a new config with another config as base, overruling a given set of parameters."""
    derived = original.copy()
    for k, v in overrules.items():
        derived[k] = v
    return derived


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
discrete_rnn = derive_config(discrete, {"model": "rnn"})
discrete_gru = derive_config(discrete, {"model": "gru"})
discrete_lstm = derive_config(discrete, {"model": "lstm"})
discrete_no_norms = derive_config(discrete, {"no_state_norming": True,
                                             "no_reward_norming": True})

# CONTINUOUS DEFAULT
#   - LunarLanderContinuous

continuous = make_config(
    batch_size=64,
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

continuous_rnn = derive_config(continuous, {"model": "rnn"})
continuous_gru = derive_config(continuous, {"model": "gru"})
continuous_lstm = derive_config(continuous, {"model": "lstm"})
continuous_beta = derive_config(continuous, {"distribution": "beta"})
continuous_no_norms = derive_config(continuous, {"no_state_norming": True,
                                                 "no_reward_norming": True})

# PENDULUM (for, well, Pendulum-v0)

pendulum = derive_config(continuous, dict(
    horizon=512,
    epochs=10,
    discount=0.99,
    distribution="gaussian"
))

pendulum_beta = derive_config(pendulum, {"distribution": "beta"})

# BIPEDAL

bipedal = derive_config(continuous, dict(
    iterations=400,
))

bipedal_rnn = derive_config(bipedal, dict(model="rnn"))
bipedal_gru = derive_config(bipedal, dict(model="gru"))
bipedal_lstm = derive_config(bipedal, dict(model="lstm"))
bipedal_beta = derive_config(bipedal, dict(distribution="beta"))

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
mujoco_vc = derive_config(mujoco, {"clip_values": True})

# ROBOSCHOOL TASKS

roboschool = make_config(
    iterations=50000000 // 2048,  # 50 million timesteps
    workers=16,
    batch_size=4096,
    horizon=1024,
    c_entropy=0.0,
    lr_pi=0.0003,
    lr_schedule="exponential",  # should be a linear annealing
    epochs=15,
    clip=0.2,
    lam=0.95,
    discount=0.99,
    grad_norm=0.5,
    clip_values=False
)

# HAND ENVS

hand = make_config(
    iterations=100,
    workers=32,
    batch_size=4096,
    horizon=512,
    c_entropy=0.01,
    lr_pi=0.0003,
    lr_schedule="exponential",
    epochs=15,
    clip=0.2,
    lam=0.95,
    discount=0.998,
    grad_norm=0.5,
    clip_values=False
)

# RECOMMENDED CONFIGS FOR ENVs

recommended_config = dict(
    **dict.fromkeys(
        ["CartPole-v2", "CartPole-v1"], discrete_no_norms
    ), **dict.fromkeys(
        ["Acrobot-v1", "MountainCar-v0"], discrete
    ), **dict.fromkeys(
        ["BipedalWalker-v2", "BipedalWalkerHardcore-v2"], bipedal
    ), **dict.fromkeys(
        ["Pendulum-v0"], pendulum_beta
    ), **dict.fromkeys(
        ["LunarLanderContinuous-v2"], continuous_beta
    ),
)
