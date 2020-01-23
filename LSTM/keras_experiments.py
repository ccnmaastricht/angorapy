#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, LSTM
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from utilities.model_management import build_sub_model_to


# from analysis.chiefinvestigation import

class Stockpredictor:

    def __init__(self):
        pass

    def model_builder(name, batch_shape, ):
        inputs = tf.keras.Input(shape=batch_shape, name="input")

        x = tf.keras.layers.LSTM(16, name='lstm')(inputs)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(4, activation="relu", name="dense")(x)

        model = tf.keras.Model(inputs=inputs, outputs=x, name=name)
        model.summary()

        return model

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

    model = Stockpredictor.model_builder("NVDA_model", (10, 4))

    samples, y, test_samples, y_test = data_processor()

    model.compile(optimizer="rmsprop", loss="mse",
                  metrics=['accuracy'])

    history = model.fit(tf.convert_to_tensor(samples, dtype=tf.float32),
                        tf.convert_to_tensor(y, dtype=tf.float32), epochs=100)

    accuracy = model.evaluate(tf.convert_to_tensor(test_samples,dtype=tf.float32),
                              tf.convert_to_tensor(y_test, dtype=tf.float32))
    print(accuracy)

    predictions = np.squeeze(model.predict(tf.convert_to_tensor(test_samples,
                                                                dtype=tf.float32)))
    plt.plot(list(range(len(predictions))), y_test[:, 0], label="Act")
    plt.plot(list(range(len(predictions))), predictions[:, 0], label="Pred")
    plt.legend()
    plt.show()

    # analysis of model
    weights = model.get_layer('lstm').get_weights()
    sub_model = build_sub_model_to(model, ['input', 'lstm'])
    sub_model.summary()

    activations = sub_model.predict(tf.convert_to_tensor(test_samples, dtype=tf.float32))
    print(activations)
    # minimize stuff -> write minimization function for lstm
