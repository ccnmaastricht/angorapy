import pickle
import matplotlib.pyplot as plt
def plot_history(retrained_history):
    """Plot the history of a set of trained RNNs"""
    plt.plot(range(len(retrained_history.history.epoch)), retrained_history.history['loss'])

    history = pickle.load(open('firsttrainhistory', "rb"))
    plt.plot(range(len(history['loss'])), history['loss'], 'r--')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['2000 iterations pretraining',
                'naive training'], loc='upper right')
    plt.show()