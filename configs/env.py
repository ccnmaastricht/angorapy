#!/usr/bin/env python
"""Configurations of hyperparameters etc. for specific gym environments.
Format is ALGORITHM -> ENV -> ID -> SETTINGS.
"""

CONFIG = {
    "PPO": {
        "CartPole": {
            "BEST": {
                "ITERATIONS": 1000,
                "AGENTS": 32,
                "EPOCHS": 3,
                "BATCH_SIZE": 32,

                "LEARNING_RATE": 0.005,
                "DISCOUNT_FACTOR": 0.99,
                "EPSILON_CLIP": 0.2
            }
        },

        "LunarLander": {
            "BEST": {
                "ITERATIONS": 1000,
                "AGENTS": 32,
                "EPOCHS": 3,
                "BATCH_SIZE": 32,

                "LEARNING_RATE": 0.005,
                "DISCOUNT_FACTOR": 0.99,
                "EPSILON_CLIP": 0.2
            }
        },

    }
}