#!/usr/bin/env python
import random

import gym
from gym.wrappers import Monitor

env = gym.make("CartPole-v1")
env = Monitor(env, "./docs/", video_callable=lambda episode_id: True, force = True)

s = env.reset()
done = False
while not done:
    s, r, done, _ = env.step(random.choice(list(range(2))))

env.close()