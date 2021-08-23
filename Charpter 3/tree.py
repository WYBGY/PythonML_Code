import math
import numpy as np
import matplotlib.pyplot as plt
import copy
import re
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns

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



