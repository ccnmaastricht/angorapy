"""Predefined configurations for different (groups of) environments."""


def make_config(batch_size=128, c_entropy=0.01, c_value=1, clip=0.2, cpu=False, debug=False, discount=0.99,
                env='CartPole-v1', epochs=3, eval=False, export_file=None, grad_norm=0.5, horizon=1024,
                iterations=1000, lam=0.97, load_from=None, lr_pi=0.001, clip_values=False, save_every=0,
                workers=8, tbptt: int = 16, lr_schedule=None, no_state_norming=False, no_reward_norming=False,
                model="ffn", early_stopping=False):
    """Make a config from scratch."""
    return dict(**locals())


def derive_config(original: dict, overrules: dict):
    """Make a new config with another config as base, overruling a given set of parameters."""
    derived = original.copy()
    for k, v in overrules.items():
        derived[k] = v
    return derived


discrete = make_config(
    batch_size=32,
    horizon=1024,
    c_entropy=0.01,
    lr_pi=0.001,
    epochs=10,
    clip=0.1,
    lam=0.95,
    discount=0.99
)

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

beta = make_config(
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

bipedal = make_config(
    batch_size=32,
    horizon=2048,
    c_entropy=0.0,
    lr_pi=0.0001,
    epochs=10,
    clip=0.2,
    lam=0.95,
    discount=0.99,
    grad_norm=0.5,
    iterations=1000,
    workers=8,
    clip_values=False
)

mujoco = make_config(
    iterations=1000000//2048,   # one million timesteps
    workers=1,
    batch_size=64,
    horizon=2048,
    c_entropy=0.0,
    lr_pi=0.0003,
    lr_schedule="exponential",
    epochs=10,
    clip=0.2,
    lam=0.95,
    discount=0.99,
    grad_norm=0.5,
    clip_values=True
)

roboschool = make_config(
    iterations=1000000//2048,   # one million timesteps
    workers=32,
    batch_size=4096,
    horizon=512,
    c_entropy=0.0,
    lr_pi=0.0003,
    lr_schedule="exponential",
    epochs=15,
    clip=0.2,
    lam=0.95,
    discount=0.99,
    grad_norm=0.5,
    clip_values=False
)

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


env_to_default_config_mapping = dict(
    **dict.fromkeys(["Humanoid-v2", "HumanoidStandup-v2"])
)
