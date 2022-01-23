from sklearn.datasets import load_iris
from sklearn.decomposition import KernelPCA, PCA
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

'''
author: heucoder
email: 812860165@qq.com
date: 2019.6.13
'''


def sigmoid(x, coef = 0.25):
    x = np.dot(x, x.T)
    return np.tanh(coef*x+1)

def linear(x):
    x = np.dot(x, x.T)
    return x

def rbf(x, gamma = 15):
    sq_dists = pdist(x, 'sqeuclidean')
    mat_sq_dists = squareform(sq_dists)
    return np.exp(-gamma*mat_sq_dists)

def kpca(data, n_dims=2, kernel = rbf):
    '''
    :param data: (n_samples, n_features)
    :param n_dims: target n_dims
    :param kernel: kernel functions
    :return: (n_samples, n_dims)
    '''

    K = kernel(data)
    #
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    #
    eig_values, eig_vector = np.linalg.eig(K)
    eig_values = eig_values.astype(float)
    eig_vector = eig_vector.astype(float)
    idx = eig_values.argsort()[::-1]
    eigval = eig_values[idx][:n_dims]
    eigvector = eig_vector[:, idx][:, :n_dims]
    print(eigval)
    eigval = eigval**(1/2)
    vi = eigvector/eigval.reshape(-1,n_dims)
    data_n = np.dot(K, vi)
    return data_n


from sklearn.datasets import make_moons
x2, y2 = make_moons(n_samples=100, random_state=123)
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(14, 6))

ax[0].scatter(x2[y2 == 0, 0], x2[y2 == 0, 1], color='red', marker='^', alpha=0.5)
ax[0].scatter(x2[y2 == 1, 0], x2[y2 == 1, 1], color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')

ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')


pca = PCA(n_components=1)
pca_data = pca.fit_transform(x2)
ax[1].scatter(pca_data[y2==0, 0], np.zeros((50, 1)), color='red', marker='^', alpha=0.5)
ax[1].scatter(pca_data[y2==1, 0], np.zeros((50, 1)), color='blue', marker='o', alpha=0.5)


my_kpca = kpca(x2, n_dims=2)
ax[2].scatter(my_kpca[y2==0, 0],  my_kpca[y2 == 0, 1], color='red', marker='^', alpha=0.5)
ax[2].scatter(my_kpca[y2==1, 0],  my_kpca[y2 == 1, 1], color='blue', marker='o', alpha=0.5)


model = KernelPCA(n_components=2, kernel='rbf', gamma=15)
kpca_x = model.fit_transform(x2)
ax[3].scatter(kpca_x[y2==0, 0],  kpca_x[y2 == 0, 1], color='red', marker='^', alpha=0.5)
ax[3].scatter(kpca_x[y2==1, 0],  kpca_x[y2 == 1, 1], color='blue', marker='o', alpha=0.5)


