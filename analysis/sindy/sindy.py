import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import rainbow
import numpy as np
from scipy.integrate import odeint
from scipy.io import loadmat

import pysindy as ps


# Generate training data
def lorenz(x, t):
    return [
        10 * (x[1] - x[0]),
        x[0] * (28 - x[2]) - x[1],
        x[0] * x[1] - 8 / 3 * x[2],
    ]

# Fit the models and simulate

poly_order = 5
threshold = 0.05
seed = 100
np.random.seed(seed)  # Seed for reproducibility

noise_level = 1e-3

models = []

x_sim = []

dt_levels = [1e-4, 1e-3, 1e-2, 0.1, 1]
for dt in dt_levels:
    t_sim = np.arange(0, 20, dt)
    t_train = np.arange(0, 100, dt)
    x0_train = [-8, 8, 27]
    x_train = odeint(lorenz, x0_train, t_train)
    x_dot_train_measured = np.array(
        [lorenz(x_train[i], 0) for i in range(t_train.size)]
    )
    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=threshold),
        feature_library=ps.PolynomialLibrary(degree=poly_order),
    )
    model.fit(
        x_train,
        t=dt,
        x_dot=x_dot_train_measured
        + np.random.normal(scale=noise_level, size=x_train.shape),
        quiet=True,
    )
    models.append(model)
    x_sim.append(model.simulate(x_train[0], t_sim))
    model.print()