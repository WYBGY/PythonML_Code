from sklearn import svm
from numpy import *


def img2Vector(filename):
    returnVec = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVec[0, 32*i + j] = int(lineStr[j])
    return returnVec


def loadImages(dir):
    hwLabels = []
    trainingFileList = os.listdir(dir)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2Vector('%s/%s' % (dir, fileNameStr))
    return trainingMat, hwLabels


dir = 'E:\资料\PythonML_Code\Charpter 5\data/trainingDigits'
train_dataArr, train_labelArr = loadImages(dir)
model = svm.SVC(C=200, kernel='rbf', tol=1e-4, max_iter=-1, degree=3, gamma='auto_deprecated', coef0=0, shrinking=True,
                probability=False, cache_size=200, verbose=False, class_weight=None, decision_function_shape='ovr')
"""
C: 惩罚因子，默认为1.0
tol: 容忍度阈值
max_iter: 迭代次数
kernel：核函数，包括:
                    linear(线性核)：u*v
                    poly(多项式核):(gamma * u * v + coef0)^degree
                    rbf(高斯核): exp(-gamma|u-v|^2)
                    sigmoid核: tanh(gamma*u*v + coef0)
degree: 多项式核中的维度
gamma： 核函数中的参数，默认为“auto”，选择1/n_features
coef： 多项式核和simoid核中的常数项，仅对这两个核函数有效
probability: 是否采用概率估计，默认为False
shrinking： 是否采用shrinking heuristic方法，默认为true
cache_size: 核函数的缓存大小，默认为200
verbose: 是否允许冗余输出
class_weight: 类别权重
decision_function_shape: 可以取'ovo'一对一和'ovr'一对其他
"""
model.fit(mat(train_dataArr), mat(train_labelArr).transpose())
train_score = model.score(mat(train_dataArr), mat(train_labelArr).transpose())
print('训练集上的准确率为%s'%train_score)

test_dataArr, test_labelArr = loadImages('E:\资料\PythonML_Code\Charpter 5\data\\testDigits'.format(dir))
test_score = model.score(mat(test_dataArr), mat(test_labelArr).transpose())
print('测试集上的准确率为%f' % test_score)
# 查看决策边界函数
model.decision_function(mat(train_dataArr))


