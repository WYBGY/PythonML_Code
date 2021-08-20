import math
import numpy as np
import matplotlib.pyplot as plt


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


def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
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

