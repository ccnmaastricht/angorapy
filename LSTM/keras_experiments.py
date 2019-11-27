from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Flatten, TimeDistributed
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

model = Sequential(name="NVDAmodel")
model.add(SimpleRNN(8, input_shape=(10, 4)))
model.add(Dense(4))
#model.add(AveragePooling1D())
#model.add(Flatten())


model.summary()
# import and analyse data
NVDA = pd.read_csv('/Users/Raphael/Downloads/NVDA.csv')

# NVDA.head()
# NVDA.iloc[:1000].plot(x="Date", y=["Open", "Close"])
# plt.show()

diff = NVDA["Close"]-NVDA["Open"]
num = NVDA.to_numpy()[:, 1:5]
#num = stats.zscore(num)
X_train = num[0:629,:]
X_test = num[629:,:]
print(X_train.shape)

#X_train = X_train.to_numpy()
#X_test = X_test.to_numpy()
X_train = stats.zscore(np.array(X_train.tolist()))
#X_test = stats.zscore(X_test)
# X_train = np.reshape(X_train, (, 4))
# X_test = np.reshape(X_test, (51, 4, 1))
samples = []
test_samples = []
y = []
y_test = []
for i in range(X_train.shape[0]-10):
    samples.append(np.expand_dims(X_train[i:(i+10)], axis=0))
    y.append(np.expand_dims(X_train[10+i], axis=0))
    test_samples.append(np.expand_dims(X_test[i:(i+10)], axis=0))
    y_test.append(np.expand_dims(X_test[10+i], axis=0))

samples = np.concatenate(samples, axis=0)
y = np.concatenate(y, axis=0)
test_samples = np.concatenate(test_samples, axis=0)
y_test = np.concatenate(y_test, axis = 0)


# X_train = np.expand_dims(X_train.to_numpy(), axis=1)
# X_test = np.expand_dims(X_test.to_numpy(), axis=1)

model.compile(optimizer="adam", loss="mse",
              metrics=['accuracy'])

history = model.fit(samples, y, epochs=1000)

accuracy = model.evaluate(test_samples, y_test)
print(accuracy)

predictions = np.squeeze(model.predict(test_samples))

plt.plot(list(range(len(predictions))), y_test, label="Act")
plt.plot(list(range(len(predictions))), predictions, label="Pred")
plt.legend()



#plt.plot(history.epoch, history.history["loss"])
plt.show()

