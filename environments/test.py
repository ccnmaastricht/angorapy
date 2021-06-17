from common.transformers import StateNormalizationTransformer
from common.wrappers import make_env

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    env = make_env("NRPReachAbsolute-v0", transformers=[StateNormalizationTransformer])

    state = env.reset()
    done = False
    while not done:
        o, r, d, i = env.step(env.action_space.sample())
        plt.imshow(o.vision)
        plt.show()