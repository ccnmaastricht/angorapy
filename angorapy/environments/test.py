from angorapy.common.transformers import StateNormalizationTransformer
from angorapy.common.wrappers import make_env

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    env = make_env("OpenAIManipulateApproxDiscrete-v0", transformers=[StateNormalizationTransformer])

    state = env.reset()
    env.render()
    done = False
    for j in range(10000):
        o, r, d, i = env.step(env.action_space.sample())
        if j % 10 == 0:
            env.render()


