import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor


data_x = load_boston()['data']
labels = load_boston()['target']

ss_X = StandardScaler()
ss_Y = StandardScaler()
m, n = np.shape(data_x)
data_x = ss_X.fit_transform(data_x.reshape(m, n))
labels = ss_Y.fit_transform(labels.reshape(m, 1))
# data_x = np.array([[338], [333], [328], [207], [226], [25], [179], [60], [208], [606]])
# labels = np.array([640, 633, 619, 393, 428, 27, 193, 66, 226, 1591])



w = np.array([0] * n)
b = 10


def gd(x, y, w, b):
    m, n = np.shape(x)
    hypo = 2 * (y.reshape(m) - (np.dot(w, x.T) + b))
    grad_w = -np.dot(hypo, x)/m
    grad_b = -np.sum(hypo)/m
    loss = np.sum((hypo/2) ** 2)/m
    return grad_w, grad_b, loss


eta = 0.0001
w_list = []
b_list = []
loss_list = []
for i in range(0, 1000000):
    grad_w, grad_b, loss = gd(data_x, labels, w, b)
    w = w - eta * grad_w
    w_list.append(w)
    b = b - eta * grad_b
    b_list.append(b)
    loss_list.append(loss)
    if loss < 1:
        break
    print(i)


lr = SGDRegressor()
lr.fit(data_x, labels)
from sklearn.metrics import mean_squared_error
mean_squared_error(labels, lr.predict(data_x))

lr = 1
iteration = 100000
w = np.array([0] * 13)
b = -120

w_history = []
b_history = []
loss_history = []


for i in range(iteration):
    b_grad = 0
    w_grad = np.array([0] * 13)
    for j in range(len(data_x)):
        w_grad = w_grad - 2 * (labels[j] - np.dot(w, data_x[j]) - b) * data_x[j]
        b_grad = b_grad - 2 * (labels[j] - np.dot(w, data_x[j] + b))

    b = b - lr * b_grad
    w = w - lr * w_grad

    b_history.append(b)
    w_history.append(w)
    print(i)


lr = SGDRegressor()
lr.fit(data_x, labels)
from sklearn.metrics import mean_squared_error
mean_squared_error(labels, lr.predict(data_x))

x = [[0, 0], [1, 2]]

def z(x):
    z = []
    for x_ in x:
        z.append(3*x_[0] + 4 * x_[1] + 1)
    return z
y = z(x)

data_x = np.array(x)
labels = np.array(y)