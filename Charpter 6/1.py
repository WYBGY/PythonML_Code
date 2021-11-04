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


def plot_digits(instances,images_per_row=10,**options):
    size=28
    # 每一行有一个
    image_pre_row=min(len(instances),images_per_row)
    images=[instances.reshape(size,size) for instances in instances]
#     有几行
    n_rows=(len(instances)-1) // image_pre_row+1
    row_images=[]
    n_empty=n_rows*image_pre_row-len(instances)
    images.append(np.zeros((size,size*n_empty)))
    for row in range(n_rows):
        # 每一次添加一行
        rimages=images[row*image_pre_row:(row+1)*image_pre_row]
        # 对添加的每一行的额图片左右连接
        row_images.append(np.concatenate(rimages,axis=1))
    # 对添加的每一列图片 上下连接
    image=np.concatenate(row_images,axis=0)
    plt.imshow(image,cmap=mpl.cm.binary,**options)
    plt.axis("off")
    plt.figure(figsize=(9,9))
    ###
example_images=np.r_[X[:12000:600],X[13000:30600:600],X[30600:60000:590]]
plot_digits(example_images,images_per_row=10)
plt.show()



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




