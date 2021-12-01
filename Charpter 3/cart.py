from numpy import *


def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        cur_line = line.strip().split('\t')
        flt_line = [float(example) for example in cur_line]
        dataMat.append(flt_line)
    return dataMat


def bin_split_dataSet(dataSet, feature, val):
    mat0 = dataSet[nonzero(dataSet[:, feature] >= val)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] < val)[0], :]

    return mat0, mat1


def regLeaf(dataSet):
    return mean(dataSet[:, -1])


def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]


def creat_tree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat is None:
        return val
    retTree = {}
    retTree['spInt'] = feat
    retTree['spVal'] = val
    lSet, rSet = bin_split_dataSet(dataSet, feat, val)
    retTree['left'] = creat_tree(lSet, leafType, errType, ops)
    retTree['right'] = creat_tree(rSet, leafType, errType, ops)
    return retTree


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]
    tolN = ops[1]
    if len(dataSet[:, -1].T.tolist()[0]) == 1:
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
            mat0, mat1 = bin_split_dataSet(dataSet, featIndex, splitVal)
            if shape(mat0)[0] < tolN or shape(mat1)[0] < tolN:
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = bin_split_dataSet(dataSet, bestIndex, bestValue)
    if shape(mat0)[0] < tolN or shape(mat1)[0] < tolN:
        return None, leafType(dataSet)
    return bestIndex, bestValue


filename = 'E:\资料\PythonML_Code\Charpter 3\ex0.txt'
dataSet = loadDataSet(filename)
dataSet = mat(dataSet)
tree = creat_tree(dataSet)


# 剪枝
# 判断该节点是否是叶子节点
def is_Tree(obj):
    return type(obj).__name__ == 'dict'


# 计算合并后的平均值
def getMean(tree):
    if is_Tree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if is_Tree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right'])/2


# 剪枝
def prune(tree, testData):
    if shape(testData)[0] == 0:
        return getMean(tree)
    if is_Tree(tree['left']) or is_Tree(tree['right']):
        lSet, rSet = bin_split_dataSet(testData, tree['spInt'], tree['spVal'])
        if is_Tree(tree['left']):
            tree['left'] = prune(tree['left'], lSet)
        if is_Tree(tree['right']):
            tree['right'] = prune(tree['right'], rSet)
    if not is_Tree(tree['left']) and not is_Tree(tree['right']):
        print(11111)
        lSet, rSet = bin_split_dataSet(testData, tree['spInt'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + sum(power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right'])/2
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print('merging')
            return treeMean
        else:
            return tree
    else:
        return tree


testData = loadDataSet('E:\资料\PythonML_Code\Charpter 3\ex2test.txt')
testData = mat(testData)


# 树模型

def linearSolve(dataSet):
    m, n = shape(dataSet)
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n-1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if linalg.det(xTx) == 0:
        return
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))


dataSet = mat(loadDataSet('E:\资料\PythonML_Code\Charpter 3\exp2.txt'))
import matplotlib.pyplot as plt

plt.scatter(dataSet[:, :-1].T.tolist()[0], dataSet[:, -1].T.tolist()[0])

model_tree = creat_tree(dataSet, leafType=modelLeaf, errType=modelErr)


def regTreeEval(model, inDat):
    return float(model)


def modelTreeEval(model, inDat):
    n = shape(inDat)[0]
    X = mat(ones((1, n+1)))
    X[:, 1:n+1] = inDat
    return float(X*model)


def treeForecast(tree, inData, modelEval=regTreeEval):
    if not is_Tree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if is_Tree(tree['left']):
            return treeForecast(tree['left'], inData, modelEval)
        else:
            return modelTreeEval(tree['left'], inData)
    else:
        if is_Tree(tree['right']):
            return treeForecast(tree['right'], inData, modelEval)
        else:
            return modelTreeEval(tree['right'], inData)


def createForcast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForecast(tree, mat(testData[i]), modelEval)
    return yHat



from tkinter import *


def reDraw(tolS, tolN):
    pass


def drawNewTree():
    pass


root = Tk()
Label(root, text='tolN').grid(row=1, column=0)
tolNentry = Entry(root)
tolNentry.grid(row=1, column=1)
tolNentry.insert(0, '10')
Label(root, text='tolS').grid(row=2, column=0)
tolSentry = Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0, '1.0')
Button(root, text='ReDraw', command=drawNewTree).grid(row=1, column=2, rowspan=3)

chkBtnVar = IntVar()
chkBtn = Checkbutton(root, text='Model Tree', variable=chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=3)

reDraw.rawDat = mat(loadDataSet('sine.txt'))
reDraw.testDat = arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)
reDraw(1.0, 10)
root.mainloop()

