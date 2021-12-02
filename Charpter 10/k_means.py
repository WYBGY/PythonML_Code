from numpy import *


# 定义距离计算
def cal_dist(vect_a, vect_b):
    return sqrt(sum(power(vect_a - vect_b, 2)))


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
                dist_ij = dis_meas(array(data[i, :]), array(centroids[j, :]))
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


data = loadData('F:\自学2020\PythonML_Code\Charpter 10\\testSet.txt')
centroids, cluster_ass = Kmeans(data, 4, dis_meas=cal_dist, create_center=rand_center)

import matplotlib.pyplot as plt


data_0 = data[nonzero(cluster_ass[:, 0].A == 0)[0]]
data_1 = data[nonzero(cluster_ass[:, 0].A == 1)[0]]
data_2 = data[nonzero(cluster_ass[:, 0].A == 2)[0]]
data_3 = data[nonzero(cluster_ass[:, 0].A == 3)[0]]
plt.scatter(data_0[:, 0].A[:, 0], data_0[:, 1].A[:, 0])
plt.plot(centroids[0, 0], centroids[0, 1], '*', markersize=20)
plt.scatter(data_1[:, 0].A[:, 0], data_1[:, 1].A[:, 0])
plt.plot(centroids[1, 0], centroids[1, 1], '*', markersize=20)
plt.scatter(data_2[:, 0].A[:, 0], data_2[:, 1].A[:, 0])
plt.plot(centroids[2, 0], centroids[2, 1], '*', markersize=20)
plt.scatter(data_3[:, 0].A[:, 0], data_3[:, 1].A[:, 0])
plt.plot(centroids[3, 0], centroids[3, 1], '*', markersize=20)



def find_best_cluster_num(sse):
    s = preprocessing.MinMaxScaler()
    sse_min_max_scalar = s.fit_transform(np.mat(sse).transpose())
    diff_sse = np.diff(sse_min_max_scalar.reshape(len(sse_min_max_scalar)), 1)
    descend_v = [(diff_sse[i - 1] / diff_sse[i]) for i in range(1, len(diff_sse) - 1)]
    best_idx = np.argmax(descend_v) + 2
    return best_idx


def bi_kmeans(data, k, dist_measure=cal_dist):
    m = shape(data)[0]
    cluster_ass = mat(zeros((m, 2)))
    # 初始化聚类中心，此时聚类中心只有一个，因此对数据取平均
    centroid0 = mean(data, axis=0).tolist()[0]
    # 存储每个簇的聚类中心的列表
    centList = [centroid0]
    for j in range(m):
        cluster_ass[j, 1] = dist_measure(mat(centroid0), data[j, :]) ** 2

    while (len(centList)) < k:
        lowestSSE = inf
        for i in range(len(centList)):
            # 在当前簇中的样本点
            point_in_current_cluster = data[nonzero(cluster_ass[:, 0].A == i)[0], :]
            # 在当前簇运用kmeans算法，分为两个簇，返回簇的聚类中心和每个样本点距离其所属簇的中心的距离
            centroid_mat, split_cluster_ass = Kmeans(point_in_current_cluster, 2, dist_measure)
            # 计算被划分的簇，划分后的损失
            sse_split = sum(split_cluster_ass[:, 1])
            # 计算没有被划分的其它簇的损失
            sse_not_split = sum(cluster_ass[nonzero(cluster_ass[:, 0].A != i)[0], 1])
            # 选择最小的损失的簇，对其进行划分
            if sse_split + sse_not_split < lowestSSE:
                # 第i个簇被划分
                best_cent_to_split = i
                # 第i个簇被划分后的聚类中心
                best_new_centers = centroid_mat
                # 第i个簇的样本，距离划分后所属的类别（只有0和1）以及距离聚类中心的距离
                best_cluster_ass = split_cluster_ass
                lowestSSE = sse_split + sse_not_split
        #
        best_cluster_ass[nonzero(best_cluster_ass[:, 0].A == 1)[0], 0] = len(centList)
        best_cluster_ass[nonzero(best_cluster_ass[:, 0].A == 0)[0], 0] = best_cent_to_split

        centList[best_cent_to_split] = best_new_centers[0, :]
        centList.append(best_new_centers[1, :])
        cluster_ass[nonzero(cluster_ass[:, 0].A == best_cent_to_split[0]), :] = best_cluster_ass

    return mat(centList), cluster_ass


