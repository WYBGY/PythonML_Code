from sklearn.datasets import fetch_openml
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl


data_x, data_y = fetch_openml('mnist_784', version=1, return_X_y=True)


def plot_digit(data):
    data = data.reshape(28, 28)
    plt.imshow(data, cmap=mpl.cm.binary, interpolation='nearest')


def array_to_imag(array):
    array = array*255
    new_image = Image.fromarray(array.astype(np.uint8))
    return new_image


def comb_imgs(origin_imgs, col, row, each_width, each_height, new_type):
    new_img = Image.new(new_type, (col * each_width, row * each_height))
    for i in range(len(origin_imgs)):
        each_img = array_to_imag(np.array(origin_imgs[i]).reshape(each_width, each_width))
        # 第二个参数为每次粘贴起始点的横纵坐标。在本例中，分别为（0，0）（28，0）（28*2，0）依次类推，第二行是（0，28）（28，28），（28*2，28）类推
        new_img.paste(each_img, ((i % col) * each_width, int((i / col)) * each_width))
    return new_img

origin_imgs = np.array(data_x.iloc[:100].values.tolist())


ten_origin_7_imgs = comb_imgs(origin_imgs, 10, 10, 28, 28, 'L')
ten_origin_7_imgs.show()


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


data_x = np.array(data_x.values.tolist())
example_images = []
for idx in range(100):
    example_images.append(data_x[idx])

plot_digits(example_images, image_per_row=10)


def pca(data, top_n_feature=9999):
    """
    :param data: 原始数据
    :param top_n_feature: 保留前n个特征
    :return: 降维后的数据和变换矩阵w
    """

    # 去中心化，减去每一列的均值
    mean_vals = data.mean(axis=0)
    mean_std = data - mean_vals

    # 计算X的协方差矩阵
    cov_x = np.cov(mean_std, rowvar=0)

    # 计算特征值，及其对应的特征向量,返回的特征值有784个，特征向量784*784
    eig_vals, eig_vectors = np.linalg.eig(np.mat(cov_x))
    eig_vals = eig_vals.astype(float)
    eig_vectors = eig_vectors.astype(float)


    # 对特征值进行排序,返回索引
    eig_vals_idx = np.argsort(eig_vals)

    # 找出前n大特征值的索引
    eig_vals_idx_max_n = eig_vals_idx[:-(top_n_feature + 1): -1]

    # 找到前n大特征值对应的特征向量, 一个特征向量是1列，返回维度（784, top_n）
    eig_vals_max_vec = eig_vectors[:, eig_vals_idx_max_n]

    # 前n个特征向量为w，对高维数据进行降维z=wx, z = (100, 784) * (784, top_n)
    new_data = mean_std * eig_vals_max_vec

    # 数据的重构，根据前n个特征重构回原数据 (100, top_n) * (top_n, 784) + (784)
    data_reconstruction = (new_data * eig_vals_max_vec.T) + mean_vals

    return new_data, eig_vals_max_vec, data_reconstruction


data = data_x[:100, :]
new_data, eig_vals_max_vec, data_reconstruction = pca(data, 40)

plt.figure()
plot_digits(data, image_per_row=10)


plt.figure()
plot_digits(data_reconstruction, image_per_row=10)


plt.figure()
plot_digits(eig_vals_max_vec.T, image_per_row=10)


from sklearn import decomposition
model = decomposition.PCA(n_components=40)
"""
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
参数：
    n_components: 可以指定降维后的维度数目，此时为大于等于1的整数；
                  也可以是指定主成分的方差和所占的最小比例阈值，让PCA根据样本方差决定降维的维数，此时参数为(0, 1];
                  也可以指定参数为"mle"，此时PCA会根据MLE算法根据方差特征的分布情况，自主选择一定的主成分数量;
                  当不输入任何时，即默认n_components=min(n_examples, n_features).
    copy: bool，True或者False，缺省时默认为True。表示是否在运行算法时，将原始训练数据复制一份。
    whiten: bool，True或者False， 是否对降维后的每一维特征进行归一化。
属性：
    components_: 降维所用的转换矩阵W，即主轴方向
    explained_variance_: 降维后各主成分的方差值，方差越大，越是主成分；
    explained_variance_ratio_: 降维后的主成分的方差值占总方差的比例；
方法：
    fit(X): 用X训练模型；
    fit_transform(X): 用X训练模型，并在X上进行降维；
    transform(X): 对X进行降维，前提要先fit；
    inverse_transform: 将降维后的数据返回原来的空间；
    
"""
mew_data = model.fit_transform(data)
transform_matrix = model.components_

