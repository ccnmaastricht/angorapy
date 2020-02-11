import sklearn.decomposition as skld
from mayavi import mlab
import matplotlib.pyplot as plt
import numpy as np


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

    def extract_fixed_point_locations(fps):
        """Processing of minimisation results for pca. The function takes one fixedpoint object at a time and
        puts all coordinates in single array."""
        fixed_point_location = []
        for fp in fps:
            fixed_point_location.append(fp['x'])
        fixed_point_locations = np.vstack(fixed_point_location)
        return fixed_point_locations

    def classify_fixedpoints(fps, unique_jac):

        # somehow this block of code does not return values if put in function

        x_modes = []
        x_directions = []
        scale = 2
        for n in range(len(unique_jac)):

            trace = np.matrix.trace(unique_jac[n]['jac'])
            det = np.linalg.det(unique_jac[n]['jac'])

            if det < 0:
                print('saddle_point')
                x_modes.append(True)
                e_val, e_vecs = np.linalg.eig(unique_jac[n]['jac'])
                ids = np.argwhere(np.real(e_val) > 0)
                for i in range(len(ids)):
                    x_plus = unique_jac[n]['x'] + scale * e_val[ids[i]] * np.real(e_vecs[:, ids[i]].transpose())
                    x_minus = unique_jac[n]['x'] - scale * e_val[ids[i]] * np.real(e_vecs[:, ids[i]].transpose())
                    x_direction = np.vstack((x_plus, unique_jac[n]['x'], x_minus))
                    x_directions.append(np.real(x_direction))
            elif det > 0:
                if trace ** 2 - 4 * det > 0 and trace < 0:
                    print('stable fixed point was found.')
                    x_modes.append(False)
                    e_val, e_vecs = np.linalg.eig(unique_jac[n]['jac'])
                    ids = np.argwhere(np.real(e_val) > 0)
                    for i in range(len(ids)):
                        x_plus = unique_jac[n]['x'] + scale * e_val[ids[i]] * np.real(e_vecs[:, ids[i]].transpose())
                        x_minus = unique_jac[n]['x'] - scale * e_val[ids[i]] * np.real(e_vecs[:, ids[i]].transpose())
                        x_direction = np.vstack((x_plus, unique_jac[n]['x'], x_minus))
                        x_directions.append(np.real(x_direction))
                elif trace ** 2 - 4 * det > 0 and trace > 0:
                    print('unstable fixed point was found')
                else:
                    print('center was found.')
                    x_modes.append(False)
            else:
                print('fixed point manifold was found.')

    def compute_unstable_modes(self):
        pass

    fixedpoints = extract_fixed_point_locations(fixedpoints)
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