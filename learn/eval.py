import statistics

import numpy

from agent.core import _RLAgent


def evaluate(agent: _RLAgent, env, episodes, show_render=False):
    scores = []
    for ep in range(episodes):
        done = False
        state = numpy.reshape(env.reset(), [1, -1])
        episode_reward = 0
        while not done:
            if ep == episodes - 1 and show_render:
                env.render()
            action = agent.act(state, )
            observation, reward, done, _ = env.step(action)
            episode_reward += reward

            state = numpy.reshape(observation, [1, -1])

        scores.append(episode_reward)

    print(f"Evaluation over {episodes} episodes: Avg. Score: {statistics.mean(scores)}\n")

