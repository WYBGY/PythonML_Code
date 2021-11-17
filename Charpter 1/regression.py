from numpy import *


def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def lwlr(testpoint, x, y, k=1.0):
    x_mat = mat(x)
    y_mat = mat(y).T
    m = shape(x)[0]
    weights = mat(eye(m))
    for i in range(m):
        diff_ = testpoint - x_mat[j, :]
        weights[i, i] = exp(diff_ * diff_.T/(-2 * k ** 2))

    xTx = x_mat.T * (weights * x_mat)
    if linalg.det(xTx) == 0:
        return
    ws = xTx.T * (x_mat.T * (weights * y_mat))
    return testpoint * ws


def lwlrTest(testArr, x, y, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros((m))
    for i in range(m):
        yHat[i] = lwlr(testArr[i], x, y, k)
    return yHat








