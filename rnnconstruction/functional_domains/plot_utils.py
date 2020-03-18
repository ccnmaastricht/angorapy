import sklearn.decomposition as skld
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_functional_domains(activations, partial_activations):

    activations = np.vstack(activations)
    pca = skld.PCA(3)
    pca.fit(activations)
    X_pca = pca.transform(partial_activations)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2])
    plt.ylim((-2, 2)), plt.xlim((-2, 2))
    ax.set_zlim(-2, 2)

    plt.show()