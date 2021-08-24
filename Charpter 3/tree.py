import math
import numpy as np
import matplotlib.pyplot as plt
import copy
import re
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
sns.set(color_codes=True)
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import missingno as msno_plot

def calcShannonEnt(data):
    num = len(data)
    # 保存每个类别的数目
    labelCounts = {}
    # 每一个样本
    for featVec in data:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # 计算信息增益
    shannonEnt = 0
    for key in labelCounts.keys():
        prob = float(labelCounts[key] / num)
        shannonEnt -= prob * math.log(prob)
    return shannonEnt


def splitData(dataSet, axis, value):
    """
    axis为某一特征维度
    value为划分该维度的值
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 舍弃掉这一维度上对应的值，剩余部分作为新的数据集
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

data = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
labels = ['no surfacing', 'flippers']

# 选择最好的特征进行数据划分
# 输入dataSet为二维List
def chooseBestFeatuerToSplit(dataSet):
    # 计算样本所包含的特征数目
    numFeatures = len(dataSet[0]) - 1
    # 信息熵H(Y)
    baseEntropy = calcShannonEnt(dataSet)
    # 初始化
    bestInfoGain = 0; bestFeature = -1
    # 遍历每个特征，计算信息增益
    for i in range(numFeatures):
        # 取出对应特征值，即一列数据
        featList = [example[i] for example in dataSet]
        uniqueVals = np.unique(featList)
        newEntropy = 0
        for value in uniqueVals:
            subDataSet = splitData(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 计算信息增益G(Y, X) = H(Y) - sum(H(Y|x))
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 按统计个数进行倒序排序
    sortedClassCount = sorted(classCount.items(), key=lambda item: item[1], reverse=True)
    return sortedClassCount[0][0]


def creatTree(dataSet, labels):
    """
    labels为特征的标签列表
    """
    classList = [example[-1] for example in dataSet]
    # 如果data中的都为同一种类别，则停止，且返回该类别
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果数据集中仅剩类别这一列了，即特征使用完，仍没有分开，则投票
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatuerToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    # 初始化树，用于存储树的结构，是很多字典的嵌套结构
    myTree = {bestFeatLabel: {}}
    # 已经用过的特征删去
    del (labels[bestFeat])
    # 取出最优特征这一列的值
    featVals = [example[bestFeat] for example in dataSet]
    # 特征的取值个数
    uniqueVals = np.unique(featVals)
    # 开始递归分裂
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = creatTree(splitData(dataSet, bestFeat, value), subLabels)
    return myTree


decisionNode = dict(boxstyle='sawtooth', fc="0.8")
leafNode = dict(boxstyle='round4', fc="0.8")
arrow_args = dict(arrowstyle='<-')


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    # plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    # plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in list(secondDict.keys()):
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in list(secondDict.keys()):
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   ]
    return listOfTrees[i]


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in list(secondDict.keys()):
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


def classify(inputTree, featLabels, testVec):
    # 自上而下搜索预测样本所属类别
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in list(secondDict.keys()):
        # 按照特征的位置确定搜索方向
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                # 若下一级结构还是dict，递归搜索
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


fr = open('E:\资料\machinelearninginaction\Ch03\lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
lenses_Tree = creatTree(lenses, lenses_labels)
createPlot(lenses_Tree)


def postPruningTree(inTree, dataSet, test_data, labels):
    """
    :param inTree: 原始树
    :param dataSet:数据集
    :param test_data:测试数据，用于交叉验证
    :param labels:标签集
    """
    firstStr = list(inTree.keys())[0]
    secondDict = inTree[firstStr]
    classList = [example[-1] for example in dataSet]
    labelIndex = labels.index(firstStr)
    temp_labels = copy.deepcopy(labels)
    del (labels[labelIndex])
    for key in list(secondDict.keys()):
        if type(secondDict[key]).__name__ == 'dict':
            if type(dataSet[0][labelIndex]).__name__ == 'str':
                subDataSet = splitData(dataSet, labelIndex, key)
                subDataTest = splitData(test_data, labelIndex, key)
                if len(subDataTest) > 0:
                    inTree[firstStr][key] = postPruningTree(secondDict[key], subDataSet, subDataTest, copy.deepcopy(labels))
    if testing(inTree, test_data, temp_labels) < testingMajor(majorityCnt(classList), temp_labels):
        return inTree
    return majorityCnt(classList)


def testing(myTree, data_test, labels):
    error = 0.0
    for i in range(len(data_test)):
        classLabel = classify(myTree, labels, data_test[i])
        if classLabel != data_test[i][-1]:
            error += 1
    return float(error)


# 测试投票节点正确率
def testingMajor(major, data_test):
    error = 0.0
    for i in range(len(data_test)):
        if major[0] != data_test[i][-1]:
            error += 1
    # print 'major %d' %error
    return float(error)

# 读取红酒数据
wine_df =pd.read_csv('F:\自学2020\PythonML_Code\Charpter 3\winequality-red.csv', sep=';')
# 查看数据， 数据有11个特征，类别为quality
wine_df.describe().transpose().round(2)
plt.title('Non-missing values by columns')
msno_plot.bar(wine_df)
# 通过箱型图查看每一列的箱型图
plt.figure()
pos = 1
for i in wine_df.columns:
    plt.subplot(3, 4, pos)
    sns.boxplot(wine_df[i])
    pos += 1

# 处理缺失值
columns_name = list(wine_df.columns)
for name in columns_name:
    q1, q2, q3 = wine_df[name].quantile([0.25, 0.5, 0.75])
    IQR = q3 - q1
    lower_cap = q1 - 1.5*IQR
    upper_cap = q3 + 1.5*IQR
    wine_df[name] = wine_df[name].apply(lambda x: upper_cap if x > upper_cap else (lower_cap if (x<lower_cap) else x))

sns.pairplot(wine_df, hue='quality')

plt.figure(figsize=(10, 8))
sns.heatmap(wine_df.corr(), annot=True, linewidths=.5, center=0, cbar=False, cmap='YlGnBu')

plt.figure(figsize=(10, 8))
sns.countplot(wine_df['quality'])

wine_df = wine_df[wine_df['quality'] != 3.5]
wine_df = wine_df[wine_df['quality'] != 7.5]
wine_df['quality'] = wine_df['quality'].replace(8, 7)
wine_df['quality'] = wine_df['quality'].replace(3, 5)
wine_df['quality'] = wine_df['quality'].replace(4, 5)
wine_df['quality'].value_counts(normalize=True)

X_train, X_test, Y_train, Y_test = train_test_split(wine_df.drop(['quality'], axis=1), wine_df['quality'], test_size=0.3, random_state=22)
print(X_train.shape, X_test.shape)

model = DecisionTreeClassifier(criterion='gini', random_state=100, max_depth=3, min_samples_leaf=5)
"""
criterion:度量函数，包括gini、entropy等
class_weight:样本权重，默认为None，也可通过字典形式制定样本权重，如：假设样本中存在4个类别，可以按照 [{0: 1, 1: 1}, {0: 1, 1: 5}, 
             {0: 1, 1: 1}, {0: 1, 1: 1}] 这样的输入形式设置4个类的权重分别为1、5、1、1，而不是 [{1:1}, {2:5}, {3:1}, {4:1}]的形式。
             该参数还可以设置为‘balance’，此时系统会按照输入的样本数据自动的计算每个类的权重，计算公式为：n_samples/( n_classes*np.bincount(y))，
             其中n_samples表示输入样本总数，n_classes表示输入样本中类别总数，np.bincount(y) 表示计算属于每个类的样本个数，可以看到，
             属于某个类的样本个数越多时，该类的权重越小。若用户单独指定了每个样本的权重，且也设置了class_weight参数，则系统会将该样本单独指定
             的权重乘以class_weight指定的其类的权重作为该样本最终的权重。
max_depth: 设置树的最大深度，即剪枝，默认为None，通常会限制最大深度防止过拟合一般为5~20，具体视样本分布来定
splitter: 节点划分策略，默认为best，还可以设置为random，表示最优随机划分，一般用于数据量较大时，较小运算量
min_sample_leaf: 指定的叶子结点最小样本数，默认为1，只有划分后其左右分支上的样本个数不小于该参数指定的值时，才考虑将该结点划分也就是说，
                 当叶子结点上的样本数小于该参数指定的值时，则该叶子节点及其兄弟节点将被剪枝。在样本数据量较大时，可以考虑增大该值，提前结束树的生长。
random_state: 当splitter设置为random时，可以通过该参数设计随机种子数
min_sample_split: 对一个内部节点划分时，要求该结点上的最小样本数，默认为2
max_features: 划分节点时，所允许搜索的最大的属性个数，默认为None，auto表示最多搜索sqrt(n)个属性，log2表示最多搜索log2(n)个属性，也可以设置整数；
min_impurity_decrease :打算划分一个内部结点时，只有当划分后不纯度(可以用criterion参数指定的度量来描述)减少值不小于该参数指定的值，才会对该
                       结点进行划分，默认值为0。可以通过设置该参数来提前结束树的生长。
min_impurity_split : 打算划分一个内部结点时，只有当该结点上的不纯度不小于该参数指定的值时，才会对该结点进行划分，默认值为1e-7。该参数值0.25
                     版本之后将取消，由min_impurity_decrease代替。
"""
model.fit(X_train, Y_train)

from sklearn.tree import export_graphviz
from six import StringIO
import pydotplus
import graphviz
from pydot import graph_from_dot_data
from IPython.display import Image


xvar = wine_df.drop(['quality'], axis=1)
feature_cols = xvar.columns

dot_data = StringIO()
export_graphviz(model, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=feature_cols, class_names=['0', '1', '2'])
(graph,) = graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
graph.write_png(r'11111.png')

dot_tree = export_graphviz(model, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=feature_cols, class_names=['0', '1', '2'])
graph = graphviz.Source(dot_data)
graph = pydotplus.graph_from_dot_data(dot_tree)



