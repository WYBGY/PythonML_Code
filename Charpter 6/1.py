import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras import Sequential
from keras.layers import Dense
from keras.layers import Activation
import matplotlib.pyplot as plt
import matplotlib as mpl


data_x, data_y = fetch_openml('mnist_784', version=1, return_X_y=True)
data_x[data_x>0] = 1
data_x = np.mat(data_x)

one_hot = OneHotEncoder()
data_y = one_hot.fit_transform(data_y.reshape(data_y.shape[0], 1)).toarray()

train_x, test_x, train_y, test_y = train_test_split(data_x, data_y)


def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=mpl.cm.binary, interpolation="nearest")
    plt.axis("off")

def


some_digit = data_x[0]
plot_digit(some_digit)


model = Sequential()
model.add(Dense(input_dim=28*28, units=500))
model.add(Activation('sigmoid'))

model.add(Dense(units=500))
model.add(Activation('sigmoid'))

model.add(Dense(units=10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_x, train_y, batch_size=100, epochs=20)




