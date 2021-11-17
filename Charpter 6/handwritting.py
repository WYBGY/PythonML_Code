from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.layers import Activation
import matplotlib.pyplot as plt
import matplotlib as mpl



data_x, data_y = fetch_openml('mnist_784', version=1, return_X_y=True)
# 将大于0的置为1，只要0和1的图片数据
data_x[data_x > 0] = 1

data_x = np.mat(data_x)

one_hot = OneHotEncoder()
data_y = one_hot.fit_transform(np.array(data_y).reshape(data_y.shape[0], 1)).toarray()

train_x, test_x, train_y, test_y = train_test_split(data_x, data_y)

model = Sequential()
model.add(Dense(input_dim=28*28, units=500))
model.add(Activation('sigmoid'))

model.add(Dense(units=500))
model.add(Activation('sigmoid'))

model.add(Dense(units=500))
model.add(Activation('sigmoid'))

for i in range(8):
    model.add(Dense(units=500))
    model.add(Activation('sigmoid'))


model.add(Dense(units=10))
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# 可选optimizer就是之前的

model.fit(train_x, train_y, batch_size=300, epochs=20)
score = model.evaluate(test_x, test_y)
print('total loss on testing data', score[0])
print('accuracy on testing data', score[1])

error_idx = []
for i in range(len(test_x)):
    predict_array = model.predict(test_x[i])
    true_array = test_y[i]
    predict_result = np.argmax(predict_array)
    true_idx = np.argwhere(true_array == 1)[0][0]
    if true_idx != predict_result:
        error_idx.append(i)


def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=mpl.cm.binary, interpolation='nearest')
    plt.axis("off")


one_digit = data_x[10000]
plot_digit(one_digit)


def plot_digits(instances, image_per_row=10, **options):
    size = 28
    image_per_row = min(len(instances), image_per_row)
    images = [instance.reshape(28, 28) for instance in instances]

    n_rows = (len(instances) - 1)//image_per_row + 1
    row_images = []
    n_empty = n_rows * image_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row*image_per_row:(row+1)*image_per_row]
        row_images.append(np.concatenate(rimages, axis=1))

    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=mpl.cm.binary, **options)
    plt.axis("off")
    plt.figure(figsize=(9, 9))


example_images = []
for idx in error_idx[:30]:
    example_images.append(test_x[idx])

plot_digits(example_images, image_per_row=10)







