import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.colors import LogNorm
from autograd import elementwise_grad, value_and_grad
import random

data_x = load_boston()['data']
labels = load_boston()['target']

# ss_X = StandardScaler()
# ss_Y = StandardScaler()
m, n = np.shape(data_x)
# data_x = ss_X.fit_transform(data_x.reshape(m, n))
# labels = ss_Y.fit_transform(labels.reshape(m, 1))
# data_x = np.array([[338], [333], [328], [207], [226], [25], [179], [60], [208], [606]])
# labels = np.array([640, 633, 619, 393, 428, 27, 193, 66, 226, 1591])


w = np.array([0] * n)
b = 0

def gd(x, y, w, b):

    hypo = 2 * (y.reshape(m) - (np.dot(w, x.T) + b))
    grad_w = -np.dot(hypo, x)/m
    grad_b = -np.sum(hypo)/m
    loss = np.sum((hypo/2) ** 2)/m
    return grad_w, grad_b, loss


eta = 0.001
w_list = []
b_list = []
loss_list = []
for i in range(0, 10000):
    grad_w, grad_b, loss = gd(data_x, labels, w, b)
    w = w - eta * grad_w
    w_list.append(w)
    b = b - eta * grad_b
    b_list.append(b)
    loss_list.append(loss)
    print(i, loss)


# f(x1, x2) = 3x1 + 4x2 + 1
x = [[0], [1], [3], [2.2], [9], [6], [4], [4.5], [5.5], [7.7]]
def f(x):
    z = []
    for x_ in x:
        z.append(3 * x_[0] + 1)
    return z

y = f(x)

data_x = np.array(x)
labels = np.array(y)

# x1_min, x1_max, x1_step = 0, 10, 0.1
# x2_min, x2_max, x2_step = 0, 10, 0.1
#
# f = lambda x1, x2: 3 * x1 + 4 * x2 + 1
# X1, X2 = np.meshgrid(np.arange(x1_min, x1_max+x1_step, x1_step), np.arange(x2_min, x2_max+x2_step, x2_step))
# Z = f(X1, X2)
# minima = np.array([0, 0])
# minima_ = minima.reshape(-1, 1)
# fig = plt.figure()
# ax = plt.axes(projection='3d', elev=50, azim=-50)
# ax.plot_surface(X1, X2, Z, norm=LogNorm(), alpha=.8, cmap=plt.get_cmap('jet'))
# ax.plot(*minima_, f(*minima_), 'r*', markersize=10)
#
# dz_dx1 = elementwise_grad(f, argnum=0)(X1, X2)
# dz_dx2 = elementwise_grad(f, argnum=1)(X1, X2)
#
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.contour(X1, X2, Z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.get_cmap('jet'))
# ax.quiver(X1, X2, X1-dz_dx1, X2-dz_dx2, alpha=0.5)
# ax.plot(*minima_, 'r*', markersize=18)


W = np.arange(0, 5, 0.1)
B = np.arange(0, 2, 0.01)
Z = np.zeros((len(W), len(B)))
for i in range(len(W)):
    for j in range(len(B)):
        w = W[i]
        b = B[j]
        Z[i][j] = 0
        for k in range(len(data_x)):
            Z[i][j] += (y[k] - w * data_x[k][0] - b) ** 2
        Z[i][j] /= len(data_x)


plt.figure()
plt.contourf(B, W, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
plt.plot([1], [3.], 'x', ms=6, marker=10, color='r')

plt.plot(b_list, w_list, 'o-', ms=3, lw=1.5, color='black')


b_grid, w_grid = np.meshgrid(B, W)
z_grid = np.zeros(np.shape(b_grid))
for i in range(len(b_grid)):
    for j in range(len(b_grid[i])):
        w = w_grid[i][j]
        b = b_grid[i][j]
        z_grid[i][j] = 0
        for k in range(len(data_x)):
            z_grid[i][j] += (y[k] - w * data_x[k][0] - b) ** 2
        z_grid[i][j] /= len(data_x)


fig = plt.figure()
ax = plt.axes(projection='3d', elev=50, azim=-50)
ax.plot_surface(b_grid, w_grid, z_grid, norm=LogNorm(), alpha=.8, cmap=plt.get_cmap('jet'))
minima = np.array([1, 3])
minima_ = minima.reshape(-1, 1)
ax.plot(*minima_, 0, 'r*', markersize=10)

ax.plot(b_list, w_list, loss_list, 'o-', ms=3, lw=1.5, color='black')


def sgd(x, y, w, b):
    idx = random.randint(0, len(x)-1)
    select_x = x[idx]
    select_y = y[idx]
    hypo = 2 * (select_y - (np.dot(w, select_x) + b))
    grad_w = -np.dot(hypo, select_x)
    grad_b = -hypo
    loss = (np.sum((y.reshape(m) - (np.dot(w, x.T) + b)) ** 2))/len(x)
    return grad_w, grad_b, loss


eta = 0.001
w_list = []
b_list = []
loss_list = []
for i in range(0, 10000):
    grad_w, grad_b, loss = sgd(data_x, labels, w, b)
    w = w - eta * grad_w
    w_list.append(w)
    b = b - eta * grad_b
    b_list.append(b)
    loss_list.append(loss)
    print(i, loss)


# adagrad
def adagrad(x, y, w, b, grad_w_list, grad_b_list):

    grad_w, grad_b, loss = gd(x, y, w, b)

    if len(grad_w_list) == 0:
        sum_grad_w = 0
        sum_grad_b = 0
    else:
        sum_grad_w = 0
        sum_grad_b = 0
        for i in range(len(grad_w_list)):
            sum_grad_w += grad_w_list[i] ** 2
            sum_grad_b += grad_b_list[i] ** 2

    sum_grad_w = sum_grad_w + grad_w**2
    sum_grad_b = sum_grad_b + grad_b**2
    delta_w = grad_w/np.sqrt(sum_grad_w)
    delta_b = grad_b/np.sqrt(sum_grad_b)
    return delta_w, delta_b, grad_w, grad_b, loss

w = np.array([0] * n)
b = 0
eta = 1
w_list = []
grad_w_list = []
b_list = []
grad_b_list = []
loss_list = []
for i in range(0, 5000):
    delta_w, delta_b, grad_w, grad_b, loss = adagrad(data_x, labels, w, b, grad_w_list, grad_b_list)
    w = w - eta * delta_w
    w_list.append(w)
    grad_w_list.append(grad_w)
    b = b - eta * delta_b
    b_list.append(b)
    grad_b_list.append(grad_b)
    loss_list.append(loss)
    print(i, loss)


def momentum(lamda, eta, w_v_list, b_v_list, grad_w_list, grad_b_list):
    if len(w_v_list) == 0:
        w_v = 0
        b_v = 0
    else:
        w_v = lamda * w_v_list[-1] - eta * grad_w_list[-1]
        b_v = lamda * b_v_list[-1] - eta * grad_b_list[-1]
    return w_v, b_v


w = np.array([0] * n)
b = 0
eta = 1
lamda = 0.01
w_list = []
grad_w_list = []
b_list = []
grad_b_list = []
loss_list = []
w_v_list = []
b_v_list = []
for i in range(0, 5000):
    grad_w, grad_b, loss = gd(data_x, labels, w, b)
    w_v, b_v = momentum(lamda, eta, w_v_list, b_v_list, grad_w_list, grad_b_list)
    grad_w_list.append(grad_w)
    grad_b_list.append(grad_b)
    w += w_v
    b += b_v
    loss_list.append(loss)
    print(i, loss)


def rmsprop(x, y, w, b, sigma_w_list, sigma_b_list, alpha):
    grad_w, grad_b, loss = gd(x, y, w, b)
    if len(sigma_w_list) == 0:
        sigma_w = grad_w
        sigma_b = grad_b
    else:
        sigma_w = np.sqrt(alpha * sigma_w_list[-1] ** 2 + (1 - alpha) * grad_w ** 2)
        sigma_b = np.sqrt(alpha * sigma_b_list[-1] ** 2 + (1 - alpha) * grad_b ** 2)
    return sigma_w, sigma_b, grad_w, grad_b, loss


w = np.array([0] * n)
b = 0
eta = 1
alpha = 0.9
w_list = []
sigma_w_list = []
b_list = []
sigma_b_list = []
loss_list = []
for i in range(0, 5000):
    sigma_w, sigma_b, grad_w, grad_b, loss = rmsprop(data_x, labels, w, b, sigma_w_list, sigma_b_list, alpha)
    w = w - eta * grad_w/sigma_w
    w_list.append(w)
    sigma_w_list.append(grad_w)
    b = b - eta * grad_b/sigma_b
    b_list.append(b)
    sigma_b_list.append(grad_b)
    loss_list.append(loss)
    print(i, loss)