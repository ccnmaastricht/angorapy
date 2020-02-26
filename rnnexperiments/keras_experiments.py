import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tensorflow as tf
from utilities.model_utils import build_sub_model_to
from fixedpointfinder.FixedPointFinder import FixedPointFinder
import sklearn.decomposition as skld
from tensorflow.keras.models import load_model
import os


class Stockpredictor:

    def __init__(self, name, batch_shape, rnn_type):
        self.model_name = name
        self.batch_shape = batch_shape
        self.rnn_type = rnn_type

        self._model_builder()
    def _model_builder(self):
        inputs = tf.keras.Input(shape=self.batch_shape, name="input")

        x = tf.keras.layers.SimpleRNN(16, name=self.rnn_type)(inputs)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(4, activation="relu", name="dense")(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=x, name=self.model_name)
        self.model.summary()

    def train(self, samples, y):
        self.model.compile(optimizer="rmsprop", loss="mse",
                              metrics=['accuracy'])

        history = self.model.fit(tf.convert_to_tensor(samples, dtype=tf.float32),
                                    tf.convert_to_tensor(y, dtype=tf.float32), epochs=80)
        self._save_model()

    def evaluate_model(self, test_samples, y_test):
        accuracy = stocker.model.evaluate(tf.convert_to_tensor(test_samples, dtype=tf.float32),
                                          tf.convert_to_tensor(y_test, dtype=tf.float32))
        print(accuracy)

    def get_weights(self):
        # analysis of model
        weights = self.model.get_layer(self.rnn_type).get_weights()

        return weights

    def get_activations(self, test_samples):
        sub_model = build_sub_model_to(self.model, ['input', self.rnn_type])
        sub_model.summary()
        activations = sub_model.predict(tf.convert_to_tensor(test_samples, dtype=tf.float32))

        return activations

    def _save_model(self):
        '''Save trained model to JSON file.'''
        self.model.save(os.getcwd()+"/saved/"+self.rnn_type+self.model_name+".h5")
        print("Saved "+self.rnn_type+" model.")

    def load_model(self):
        """Load saved model from JSON file.
        The function will overwrite the current model, if it exists."""
        self.model = load_model(os.getcwd()+"/saved/"+self.rnn_type+self.model_name+".h5")
        print("Loaded "+self.rnn_type+" model.")


def data_processor():

    # import and analyse data
    NVDA = pd.read_csv('/Users/Raphael/Downloads/NVDA.csv')

    num = NVDA.to_numpy()[:4000, 1:5]
    train_size = 0.7
    X_train = num[0:int(NVDA.shape[0]*train_size),:]
    X_test = num[-int((1-train_size)*NVDA.shape[0]):,:]
    print(X_train.shape)

    #X_train = X_train.to_numpy()
    #X_test = X_test.to_numpy()
    # X_train = stats.zscore(np.array(X_train.tolist()))
    #X_test = stats.zscore(X_test)
    # X_train = np.reshape(X_train, (, 4))
    # X_test = np.reshape(X_test, (51, 4, 1))
    samples, test_samples, y, y_test = [], [], [], []
    for i in range(X_train.shape[0]-10):
        samples.append(np.expand_dims(X_train[i:(i+10)], axis=0))
        y.append(np.expand_dims(X_train[10+i], axis=0))
    for i in range(X_test.shape[0]-10):
        test_samples.append(np.expand_dims(X_test[i:(i+10)], axis=0))
        y_test.append(np.expand_dims(X_test[10+i], axis=0))

    samples = np.concatenate(samples, axis=0)
    y = np.concatenate(y, axis=0)
    test_samples = np.concatenate(test_samples, axis=0)
    y_test = np.concatenate(y_test, axis = 0)

    return samples, y, test_samples, y_test
if __name__ == "__main__":

    stocker = Stockpredictor("NVDA_model", (10, 4), rnn_type='vanilla')

    samples, y, test_samples, y_test = data_processor()

    # stocker.train(samples, y)
    stocker.load_model()
    stocker.evaluate_model(test_samples, y_test)

    weights = stocker.get_weights()
    activations = stocker.get_activations(test_samples)

    pca = skld.PCA(3)
    pca.fit(activations[1])
    X_pca = pca.transform(activations[1])

    n_points = 1500

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(X_pca[:n_points, 0], X_pca[:n_points, 1], X_pca[:n_points, 2],
            linewidth=0.2)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.show()



    # predictions = np.squeeze(stocker.model.predict(tf.convert_to_tensor(test_samples,
                                                       #         dtype=tf.float32)))
    #plt.plot(list(range(len(predictions))), y_test[:, 0], label="Act")
    #plt.plot(list(range(len(predictions))), predictions[:, 0], label="Pred")
    #plt.legend()
    #plt.show()

