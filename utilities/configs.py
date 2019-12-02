#!/usr/bin/env python
from argparse import Namespace


def make_config(batch_size=128, c_entropy=0.01, c_value=1, clip=0.2, cpu=False, debug=False, discount=0.99,
                env='CartPole-v1', epochs=3, eval=False, export_file=None, grad_norm=0.5, horizon=1024,
                iterations=1000, lam=0.97, load_from=None, lr_pi=0.0003, no_value_clip=True, save_every=0,
                workers=4):
    """Make a config."""

    return Namespace(**locals())


discrete = make_config(
    batch_size=32,
    horizon=128,
    c_entropy=0.01,
    lr_pi=0.0003,
    epochs=10,
    clip=0.1,
    lam=0.95,
    discount=0.99
)

continuous = make_config(
    grad_norm=None
)

bipedal = make_config(
    batch_size=32,
    horizon=2048,
    c_entropy=0.0,
    lr_pi=0.0003,
    epochs=10,
    clip=0.1,
    lam=0.95,
    discount=0.99,
    grad_norm=0,
)
