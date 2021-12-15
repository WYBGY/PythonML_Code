import numpy as np
import random
import matplotlib.pyplot as plt
import copy
from sklearn import datasets


# 搜索邻域内的点
def find_neighbor(j, x, eps):
    """
    :param j: 核心点的索引
    :param x: 数据集
    :param eps:邻域半径
    :return:
    """
    temp = np.sum((x - x[j]) ** 2, axis=1) ** 0.5
    N = np.argwhere(temp <= eps).flatten().tolist()
    return N


def DBSCAN(X, eps, MinPts):
    k = -1
    # 保存每个数据的邻域
    neighbor_list = []
    # 核心对象的集合
    omega_list = []
    # 初始化，所有的点记为未处理
    gama = set([x for x in range(len(X))])
    cluster = [-1 for _ in range(len(X))]

    for i in range(len(X)):
        neighbor_list.append(find_neighbor(i, X, eps))
        if len(neighbor_list[-1]) >= MinPts:
            omega_list.append(i)

    omega_list = set(omega_list)
    while len(omega_list) > 0:
        gama_old = copy.deepcopy(gama)
        # 随机选取一个核心点
        j = random.choice(list(omega_list))
        # 以该核心点建立簇Ck
        k = k + 1
        Q = list()
        # 选取的核心点放入Q中处理，Q中只有一个对象
        Q.append(j)
        # 选取核心点后，将核心点从核心点列表中删除
        gama.remove(j)
        # 处理核心点，找出核心点所有密度可达点
        while len(Q) > 0:
            q = Q[0]
            # 将核心点移出，并开始处理该核心点
            Q.remove(q)
            # 第一次判定为True，后面如果这个核心点密度可达的点还有核心点的话
            if len(neighbor_list[q]) >= MinPts:
                # 核心点邻域内的未被处理的点
                delta = set(neighbor_list[q]) & gama
                delta_list = list(delta)
                # 开始处理未被处理的点
                for i in range(len(delta)):
                    # 放入待处理列表中
                    Q.append(delta_list[i])
                    # 将已处理的点移出标记列表
                    gama = gama - delta
        # 本轮中被移除的点就是属于Ck的点
        Ck = gama_old - gama
        Cklist = list(Ck)
        # 依次按照索引放入cluster结果中
        for i in range(len(Ck)):
            cluster[Cklist[i]] = k
        omega_list = omega_list - Ck
    return cluster


X1, y1 = datasets.make_circles(n_samples=2000, factor=.6, noise=.02)
X2, y2 = datasets.make_blobs(n_samples=400, n_features=2, centers=[[1.2, 1.2]], cluster_std=[[.1]], random_state=9)
X = np.concatenate((X1, X2))
eps = 0.08
min_Pts = 10
C = DBSCAN(X, eps, min_Pts)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=C)
plt.show()

from sklearn.cluster import DBSCAN

model = DBSCAN(eps=0.08, min_samples=10, metric='euclidean', algorithm='auto')
"""
eps: 邻域半径
min_samples：对应MinPts
metrics: 邻域内距离计算方法，之前在层次聚类中已经说过，可选有： 
        欧式距离：“euclidean”
        曼哈顿距离：“manhattan”
        切比雪夫距离：“chebyshev” 
        闵可夫斯基距离：“minkowski”
        带权重的闵可夫斯基距离：“wminkowski”
        标准化欧式距离： “seuclidean”
        马氏距离：“mahalanobis”
algorithm：最近邻搜索算法参数，算法一共有三种，
        第一种是蛮力实现‘brute’，
        第二种是KD树实现‘kd_tree’，
        第三种是球树实现‘ball_tree’， 
        ‘auto’则会在上面三种算法中做权衡
leaf_size：最近邻搜索算法参数，为使用KD树或者球树时， 停止建子树的叶子节点数量的阈值
p: 最近邻距离度量参数。只用于闵可夫斯基距离和带权重闵可夫斯基距离中p值的选择，p=1为曼哈顿距离， p=2为欧式距离。

"""
model.fit(X)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=model.labels_)
plt.show()