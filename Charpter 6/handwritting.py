from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.layers import activations




data_x, data_y = fetch_openml('mnist_784', version=1, return_X_y=True)
data_x[data_x > 0] = 1

data_x = np.mat(data_x)

one_hot = OneHotEncoder()
data_y = one_hot.fit_transform(np.array(data_y).reshape(data_y.shape[0], 1))

train_x, test_x, train_y, test_y = train_test_split(data_x, data_y)

model = Sequential()






