#!/usr/bin/env python


def make_config(batch_size=128, c_entropy=0.01, c_value=1, clip=0.2, cpu=False, debug=False, discount=0.99,
                env='CartPole-v1', epochs=3, eval=False, export_file=None, grad_norm=0.5, horizon=1024,
                iterations=1000, lam=0.97, load_from=None, lr_pi=0.001, clip_values=False, save_every=0,
                workers=8, tbptt: int = 16, lr_schedule=None):
    """Make a config."""

    return dict(**locals())


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
