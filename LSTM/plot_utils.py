import sklearn.decomposition as skld
from mayavi import mlab
import matplotlib.pyplot as plt


def plot_velocities(activations, velocities, n_points: int = 5000):
    """This functions creates a 3D line plot of network activities
    colored by 'kinetic energy' i.e. rate of change in the dynamic system at
    any given position.

    Args:
        activations: Network activations. Must have dimensions [n_time, n_units].

        velocities: Vector of 'velocities' computed by feeding activations in q(x).
        Must have dimensions [n_time].

        n_points (optional): Number of points to be used for line plot. Recommended to choose values
         between 2000 and 5000. Default: 5000.

     Returns:
         None."""
    pca = skld.PCA(3)
    pca.fit(activations)
    X_pca = pca.transform(activations)

    mlab.plot3d(X_pca[:n_points, 0], X_pca[:n_points, 1], X_pca[:n_points, 2],
                velocities[:n_points])
    mlab.colorbar(orientation='vertical')
    mlab.show()

def plot_fixed_points(self, activations, fixedpoints, n_points):
    activations = np.vstack(activations)

    pca = skld.PCA(3)
    pca.fit(activations)
    X_pca = pca.transform(activations)
    new_pca = pca.transform(fixedpoints)



    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(X_pca[:n_points, 0], X_pca[:n_points, 1], X_pca[:n_points, 2],
            linewidth=0.2)
    for i in range(len(x_modes)):
        if not x_modes[i]:
            ax.scatter(new_pca[i, 0], new_pca[i, 1], new_pca[i, 2],
                       marker='.', s=30, c='k')
        else:
            ax.scatter(new_pca[i, 0], new_pca[i, 1], new_pca[i, 2],
                       marker='.', s=30, c='r')
            for p in range(len(x_directions)):
                direction_matrix = pca.transform(x_directions[p])
                ax.plot(direction_matrix[:, 0], direction_matrix[:, 1], direction_matrix[:, 2],
                        c='r', linewidth=0.8)

    plt.title('PCA using modeltype: ' + self.hps['rnn_type'])
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.show()