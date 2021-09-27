import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.colors import LogNorm
from autograd import elementwise_grad, value_and_grad

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


eta = 0.000001
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
    pass
