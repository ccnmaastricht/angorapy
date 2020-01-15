import numpy as np

class Avemover:

    def __init__(self, N):
        self.N = N


    def gen_data(self, size):
        pass
        delta_t = 0.001
        seconds = 1000
        nPulses = 100
        x = np.random.uniform(-1, 1, nPulses)
        pulses = np.reshape(np.repeat(x, int(1/delta_t)), (nPulses, int(1/delta_t))).transpose()




