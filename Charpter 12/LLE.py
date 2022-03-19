import numpy as np
from sklearn.datasets import make_s_curve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def cal_pairwise_dist(x):
    # 输入矩阵x，返回两两之间的距离
    """
    输入矩阵x，返回两两之间的距离
    (a-b)^2 = a^2 + b^2 - 2ab
    """
    # a^2 + b^2
    sum_x = np.sum(np.square(x), axis=1)

    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    return dist


def get_n_neighbors(data, n_neighbors=10):
    dist = cal_pairwise_dist(data)
    dist[dist < 0] = 0
    dist = dist ** 0.5

    n = dist.shape[0]
    N = np.zeros((n, n_neighbors))

    for i in range(n):
        # 计算每一个样本点，距离其最近的近邻点的索引
        index_ = np.argsort(dist[i])[1:n_neighbors+1]
        # 距离每一个样本最近的点的索引i
        N[i] = N[i] + index_

    return N.astype(np.int32)


def lle(data, n_dims=2, n_neighbors=10):
    # 先获取到样本点的近邻的样本索引
    N = get_n_neighbors(data, n_neighbors)
    # 样本数量n，维数为D
    n, D = data.shape

    # 当原空间维度小于近邻点数量时，W不是满秩的，要进行特殊处理
    if n_neighbors > D:
        tol = 1e-3
    else:
        tol = 0

    # 初始化W，W应该是n * n——neighbors维度，即n个样本有n个wi，每一个wi有n_neighbors， 这里做了转置
    W = np.zeros((n_neighbors, n))
    # 即1k，k维全为1的列向量
    I = np.ones(n_neighbors, 1)

    for i in range(n):
        # 对于每一个样本点xi
        # 先将xi进行伸展，形状同xj一致
        Xi = np.tile(data[i], (n_neighbors, 1)).T
        # xj所组成的矩阵
        Ni = data[N[i]].T
        # 求Yi
        Yi = np.dot((Xi-Ni).T, (Xi - Ni))
        # 这里是对于样本维度小于n_neighbors时做的特殊处理，MLLE算法，保持局部邻域关系的增量Hessian LLE算法
        Yi = Yi + np.eye(n_neighbors) * tol * np.trace(Yi)

        # 求解逆矩阵
        Yi_inv = np.linalg.pinv(Yi)
        # 求解每一个样本的wi，并做归一化处理
        wi = (np.dot(Yi_inv, I))/(np.dot(np.dot(I.T, Yi_inv), I)[0, 0])
        W[:, i] = wi[:, 0]

    # 初始化W
    W_y = np.zeros((n, n))

    # 对上一步求的W做进一步扩充，之前是n*k维的，现在变成n*n维的，不是近邻的位置补0
    for i in range(n):
        index = N[i]
        for j in range(n_neighbors):
            W_y[index[j], i] = W[j, i]


    I_y = np.eye(n)
    # 计算(I-W)(I-W).T
    M = np.dot((I_y - W_y), (I_y - W_y).T)
    # 求特征值
    eig_val, eig_vector = np.linalg.eig(M)
    # 找出前n_dim个小的特征值，忽略掉0，取第2到第n_dim+1个
    index_ = np.argsort(np.abs(eig_val))[1: n_dims+1]
    print("index_", index_)
    # 特征值对应的特征向量就是最后降维后得到的样本
    Y = eig_vector[:, index_]

    return Y







