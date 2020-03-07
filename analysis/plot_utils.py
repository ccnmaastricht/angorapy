import matplotlib.pyplot as plt
import scipy as sp
import sklearn.decomposition as skld
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

def plot_results(self, results, action_data, fixed_point, title: str = "Results", dreiD: bool = False):
    """Plot results of analysis performed by chiefinvestigator

    Args:
        results: Results of analysis to be plotted.
        action_data: Actions selected by agent that are used to color the plots.
        title: Title of the plot. Default is set to "Results"
        :param dreiD:
        :param fixed_point:

    Returns:
        Plot of data to be visualized

    """
    if dreiD is True:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(results[:, 0], results[:, 1], results[:, 2],
                   marker='o', s=5, c=action_data[:])
        if fixed_point is not None:
            ax.scatter(fixed_point[:, 0], fixed_point[:, 1], fixed_point[:, 2],
                       marker='x', s=30)
        plt.title(title)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.show()

    if dreiD is False:
        plt.figure()
        plt.scatter(results[:, 1], results[:, 0],
                    c=action_data[:],
                    label=action_data[:])
        # legend_elements = [Line2D([0], [0], marker='o', label='Action: 0', color='w',
        #                         markerfacecolor='y', markersize=10),
        #                  Line2D([0], [0], marker='o', label='Action: 1', color='w',
        #                        markerfacecolor='tab:purple', markersize=10)]
        # plt.legend(handles=legend_elements)
        plt.title(title)
        plt.xlabel('second component')
        plt.ylabel('first component')
        # plt.pause(0.5)
        plt.show()


def plot_rewards(self, rewards):
    plt.plot(list(range(len(rewards))), rewards)
    plt.xlabel('Number of steps in episode')
    plt.ylabel('Numerical reward')
    plt.title('Reward over episode')
    plt.show()


def timestepwise_pca(self, activation_data, action_data, title: str = "Results"):
    zscores = sp.stats.zscore(activation_data)  # normalization

    pca = skld.PCA(3)
    pca.fit(zscores)
    X_pca = pca.transform(zscores)

    fig = plt.figure()
    plt.xlim([-3, 4])
    plt.ylim([-7.5, 10.5])

    def update(i):
        plot_results(X_pca[i:(i + 10), :], action_data[i:(i + 10), 0], title)  # plot pca results

    anim = animation.FuncAnimation(fig, update, frames=int(len(X_pca) - 1), interval=100)
    anim.save("moving_pca.gif",
              writer='pillow')