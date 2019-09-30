#!/usr/bin/env python
"""Configurations of hyperparameters etc. for specific gym environments.
Format is ALGORITHM -> ENV -> ID -> SETTINGS.
"""

CONFIG = {
    "PPO": {
        "CartPole-v1": {
            "BEST": {
                "ITERATIONS": 20,
                "AGENTS": 32,
                "HORIZON": 2048,
                "EPOCHS": 6,
                "BATCH_SIZE": 32,

                "LEARNING_RATE": 0.005,
                "DISCOUNT_FACTOR": 0.99,
                "EPSILON_CLIP": 0.2
            },

            "DEBUG": {
                "ITERATIONS": 10,
                "AGENTS": 3,
                "EPOCHS": 2,
                "BATCH_SIZE": 128,
                "HORIZON": 500,

                "LEARNING_RATE": 0.005,
                "DISCOUNT_FACTOR": 0.99,
                "EPSILON_CLIP": 0.2
            }
        },

        "LunarLander-v2": {
            "BEST": {
                "ITERATIONS": 1000,
                "AGENTS": 16,
                "HORIZON": 2048,
                "EPOCHS": 6,
                "BATCH_SIZE": 32,

                "LEARNING_RATE": 0.0005,
                "DISCOUNT_FACTOR": 0.99,
                "EPSILON_CLIP": 0.2
            }
        },

        "Pendulum": {
            "BEST": {
                "ITERATIONS": 30,
                "AGENTS": 32,
                "EPOCHS": 6,
                "BATCH_SIZE": 32,

                "LEARNING_RATE": 0.01,
                "DISCOUNT_FACTOR": 0.99,
                "EPSILON_CLIP": 0.2
            }
        },

        "Acrobot": {
            "BEST": {
                "ITERATIONS": 30,
                "AGENTS": 32,
                "EPOCHS": 6,
                "BATCH_SIZE": 32,

                "LEARNING_RATE": 0.01,
                "DISCOUNT_FACTOR": 0.99,
                "EPSILON_CLIP": 0.2
            }
        },

        "TunnelRAM": {
            "BEST": {
                "ITERATIONS": 100,
                "AGENTS": 32,
                "HORIZON": 2048,
                "EPOCHS": 6,
                "BATCH_SIZE": 64,

                "LEARNING_RATE": 0.01,
                "DISCOUNT_FACTOR": 0.995,
                "EPSILON_CLIP": 0.2
            }
        },

        "Pong-ram": {
            "BEST": {
                "ITERATIONS": 1000,
                "AGENTS": 32,
                "HORIZON": 2048,
                "EPOCHS": 6,
                "BATCH_SIZE": 64,

                "LEARNING_RATE": 0.0005,
                "DISCOUNT_FACTOR": 0.995,
                "EPSILON_CLIP": 0.2
            }
        },

    }
}