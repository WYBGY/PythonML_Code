from numpy import *


# 定义距离计算
def cal_dist(vect_a, vect_b):
    return sqrt(sum(power(vect_a, vect_b, 2)))


# 随机选取聚类中心
def rand_center(data, k):
    n = shape(data)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        min_j = min(data[:, j])
        range_j = float(max(data[:, j]) - min(data[:, j]))
        # np.random.rand(k, 1) 生成size为（k,1）的0~1的随机array
        centroids[:, j] = min_j + range_j * random.rand(k, 1)
    return centroids


def Kmeans(data, k, dis_meas=cal_dist, create_center=rand_center):
    m = shape(data)[0]
    # 用于保存每个样本所属类别的矩阵，第0维为所属类别，第一维为样本距离该类别的距离
    clusterAssment = mat(zeros((m, 2)))
    # 初始化聚类中心
    centroids = create_center(data, k)
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            min_dist = inf
            min_index = -1
            for j in range(k):
                dist_ij = dis_meas(data[i, :], centroids[j, :])
                if dist_ij < min_dist:
                    min_dist = dist_ij
                    min_index = j
            # 如果样本的类别发生了变化，则继续迭代
            if clusterAssment[i, 0] != min_index:
                cluster_changed = True
            # 第i个样本距离最近的中心j存入
            clusterAssment[i, :] = min_index, min_dist ** 2
        print(centroids)
        # 重新计算聚类中心
        for cent in range(k):
            # 找出数据集中属于第k类的样本的所有数据，nonzero返回索引值
            points_in_cluster = data[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(points_in_cluster, axis=0)
    return centroids, clusterAssment


def loadData(filename):
    data_mat = []
    fr = open(filename)
    for line in fr.readlines():
        cur_line = line.strip().split('\t')
        flt_line = [float(example) for example in cur_line]
        data_mat.append(flt_line)
    return mat(data_mat)



