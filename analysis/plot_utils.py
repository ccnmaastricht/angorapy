import matplotlib.pyplot as plt
import scipy as sp
import sklearn.decomposition as skld
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class Visualise:

    def __init__(self, activations, n_pcs: int = 3, multiple_activation_data: list = False):
        self.activations = activations
        self.n_pcs = n_pcs
        self.multiple_activation_data = multiple_activation_data

        self.__transform_activations__()

    def __transform_activations__(self):
        self.pca = skld.PCA(self.n_pcs)
        self.transformed_activations = self.pca.fit_transform(self.activations)

    def __inverse_transform__(self, data):
        return self.pca.inverse_transform(data)

    def plot_activations(self):
        plt.plot(self.activations)

    def plot_activations_and_q(self, q_vector):
        fig = plt.figure(1)
        plt.subplot(121)
        plt.plot(self.activations)
        plt.ylabel('Activation')
        plt.xlabel('Time')
        plt.subplot(122)
        plt.plot(q_vector)
        plt.ylabel('q value')
        plt.xlabel('Time')

        return fig

    def mesh_around_point_2d(self, index, size, n_points):
        if self.n_pcs == 3:
            inverse_transformed_mesh = self.mesh_around_point_3d(index, size, n_points)
        else:
            central_point = self.transformed_activations[index, :]

            plus_point = central_point.T + central_point.T * size
            minus_point = central_point.T - central_point.T * size

            x, y = np.meshgrid(np.linspace(minus_point[0], plus_point[0], n_points),
                               np.linspace(minus_point[1], plus_point[1], n_points),
                               indexing='xy')
            x, y = x.ravel(), y.ravel()

            stacked_points = np.vstack((x, y)).T
            inverse_transformed_mesh = self.__inverse_transform__(stacked_points)

        return inverse_transformed_mesh

    def mesh_around_point_3d(self, index, size, n_points):
        central_point = self.transformed_activations[index, :]

        plus_point = central_point.T + central_point.T * size
        minus_point = central_point.T - central_point.T * size

        x, y, z = np.meshgrid(np.linspace(minus_point[0], plus_point[0], n_points),
                           np.linspace(minus_point[1], plus_point[1], n_points),
                           np.linspace(minus_point[2], plus_point[2], n_points),
                           indexing='ij')
        x, y, z = x.ravel(), y.ravel(), z.ravel()

        stacked_points = np.vstack((x, y, z)).T
        inverse_transformed_mesh = self.__inverse_transform__(stacked_points)

        return inverse_transformed_mesh

    def plot_activations_3d(self, n_points):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot(self.activations[:n_points, 0], self.activations[:n_points, 1], self.activations[:n_points, 2],
                linewidth=0.7)
        if self.multiple_activation_data:
            for activation in self.multiple_activation_data:
                ax.plot(activation[:n_points, 0], activation[:n_points, 1], activation[:n_points, 2],
                        linewidth=0.7)

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')

        return fig

    @staticmethod
    def plot_rewards(rewards):
        plt.plot(list(range(len(rewards))), rewards)
        plt.xlabel('Number of steps in episode')
        plt.ylabel('Numerical reward')
        plt.title('Reward over episode')
        plt.show()




