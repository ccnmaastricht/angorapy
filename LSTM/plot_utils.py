import sklearn.decomposition as skld
from mayavi import mlab


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